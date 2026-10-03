//! Client-held prepared projection inputs; projection/conv/recurrence share a query.
use crate::{Manifest,Request,Result,MAX_FLOATS,prepared_weights::WeightBuffer,int8_kernel};
const COLS:usize=2560;
const RANK:usize=64;
const PREPARED:usize=COLS+COLS/256+2*RANK+64;
pub(super) fn projection(r:&Request,m:&Manifest,name:String,n:usize,rows:usize,start:usize)->Result<Request> {
    let prefix=name.strip_suffix(".weight").ok_or("Delta projection name")?;
    let mut p=r.clone();p.op="lora_integer".into();p.tensor=name.clone();p.dims=vec![n,rows,COLS,start];p.scalars=vec![2.];p.aux=vec![format!("{prefix}.lora_A.weight"),format!("{prefix}.lora_B.weight")];
    let base=m.tensors.iter().find(|t|t.name==name).ok_or("Delta projection base")?;
    let a=m.tensors.iter().find(|t|t.name==p.aux[0]).ok_or("Delta projection A")?;
    let b=m.tensors.iter().find(|t|t.name==p.aux[1]).ok_or("Delta projection B")?;
    let total=if name.ends_with(".in_proj_qkv.weight") {8192} else {4096};
    if base.dtype!="int8" || base.rows!=total || base.cols!=COLS || a.dtype!="f32" || a.rows!=RANK || a.cols!=COLS || b.dtype!="f32" || b.rows!=total || b.cols!=RANK || start.checked_add(rows).is_none_or(|v|v>total) {return Err("Delta projection weights".into());}
    Ok(p)
}
pub(super) fn a_product<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    let a=m.tensors.iter().find(|t|t.name==r.aux[0]).ok_or("Delta A")?;
    let mut ar=r.clone();ar.op="matmul".into();ar.dims=vec![r.dims[0],RANK,COLS];
    let (w,bytes)=crate::load_prepared_weight(a,&ar,&mut *read)?;
    Ok((crate::profile::measure("lora_matmul_A",||crate::matrix_loaded(x,&w,r.dims[0],RANK,COLS))?,bytes))
}
pub(super) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    if !crate::lossless_encoding(&r.encoding) || r.dims.len()!=4 || !r.aux.is_empty() || !r.scalars.is_empty() || !matches!(r.op.as_str(),"delta_project_capture"|"delta_project_reuse") || !x.iter().all(|v|v.is_finite()) {return Err("Delta projected metadata".into());}
    let (n,h,first,keep)=(r.dims[0],r.dims[1],r.dims[2],r.dims[3]);
    if n==0 || n>132 || h==0 || h>16 || h%2!=0 || first>30 || first%2!=0 || first+h>32 || keep>1 {return Err("Delta projected shape".into());}
    let capture=r.op=="delta_project_capture";let channels=h*256;let history=3*channels;let state=h*16384;let input=if capture {n*COLS} else {n*PREPARED};
    let count=n*h*128;let output=count+history+if keep==1 {state} else {0};
    if x.len()!=input+history+state || x.len()>MAX_FLOATS || output+if capture {n*PREPARED} else {0}>MAX_FLOATS {return Err("Delta projected frame bounds".into());}
    let floats=output+if capture {n*PREPARED} else {0};
    let bitmap=if r.encoding=="bf16-exact" {floats.div_ceil(8)} else {floats.div_ceil(256).div_ceil(8)};
    let header=serde_json::to_vec(r).map_err(|e|e.to_string())?.len()+1;
    // Integer input lanes are exact BF16; all scale/A/gate/state tails may be
    // full F32. Reserve for a progress digit before any expensive weight read.
    let max_bytes=4+header+32+4+bitmap+2*(count+history)+4*if keep==1 {state} else {0}+if capture {n*(COLS*2+(PREPARED-COLS)*4)} else {0};
    if max_bytes>2_000_000 {return Err("Delta projected reply byte bound".into());}
    let root=r.tensor.strip_suffix(".in_proj_qkv.weight").filter(|s|s.ends_with(".linear_attn")).ok_or("Delta projected tensor")?;
    let mut qp=projection(r,m,r.tensor.clone(),n,h/2*128,first/2*128)?;
    let zp=projection(r,m,format!("{root}.in_proj_z.weight"),n,h*128,first*128)?;
    let work=n as u64*((channels+h*128+64) as u64*COLS as u64+RANK as u64*(channels+h*128+2*COLS) as u64);
    if work>3_000_000_000 {return Err("Delta projected work limit".into());}
    let mut bytes=0;let mut saved=Vec::new();
    let (q,qa,za,gates)=if capture {
        let original=&x[..input];
        let q=crate::profile::measure("activation_quantize",||int8_kernel::quantize_rows(original,n,COLS))?;
        let (qa,used)=a_product(&qp,original,m,read)?;bytes+=used;
        let (za,used)=a_product(&zp,original,m,read)?;bytes+=used;
        let mut gr=r.clone();gr.op="delta_gates_integer".into();gr.tensor=format!("{root}.in_proj_a.weight");gr.dims=vec![n];
        let (gates,used)=crate::delta_stage::gates_cached(&gr,original,m,read,Some(&q))?;bytes+=used;
        q.append_wire_values(&mut saved);saved.extend_from_slice(&q.scales()[..n*(COLS/256)]);saved.extend_from_slice(&qa);saved.extend_from_slice(&za);saved.extend_from_slice(&gates);
        (q,qa,za,gates)
    } else {
        let integers=n*COLS;let scales=n*(COLS/256);let a=n*RANK;
        let q=crate::profile::measure("activation_restore",||int8_kernel::QuantizedRows::from_wire(n,COLS,&x[..integers],&x[integers..integers+scales]))?;
        let rest=&x[integers+scales..input];
        (q,rest[..a].to_vec(),rest[a..2*a].to_vec(),rest[2*a..].to_vec())
    };
    let original=if capture {&x[..input]} else {&[]};
    let mut mixed=vec![0.;n*channels];let mut column=0;
    for (start,rows) in [(first/2*128,h/2*128),(2048+first/2*128,h/2*128),(4096+first*128,h*128)] {
        qp.dims=vec![n,rows,COLS,start];
        let (tile,used)=crate::evaluate_integer_with_ax(&qp,original,m,read,Some(&q),Some(&qa))?;bytes+=used;
        for t in 0..n {mixed[t*channels+column..t*channels+column+rows].copy_from_slice(&tile[t*rows..(t+1)*rows]);}
        column+=rows;
    }
    let (z,used)=crate::evaluate_integer_with_ax(&zp,original,m,read,Some(&q),Some(&za))?;bytes+=used;
    let mut window=x[input..input+history].to_vec();window.extend_from_slice(&mixed);
    let final_history=window[window.len()-history..].to_vec();
    let mut stage=window;stage.extend(z);
    for begin in [0,n*32] {for t in 0..n {stage.extend_from_slice(&gates[begin+t*32+first..begin+t*32+first+h]);}}
    stage.extend_from_slice(&x[input+history..]);
    let mut dr=r.clone();dr.op="delta_stage_bf16".into();dr.tensor=format!("{root}.conv1d.weight");dr.aux=vec![format!("{root}.norm.weight")];
    let (values,used)=crate::delta_stage::evaluate_projected(&dr,&stage,m,read)?;bytes+=used;
    let mut out=values[..count].to_vec();out.extend(final_history);out.extend_from_slice(&values[count..]);if capture {out.extend(saved);}
    if out.len()!=output+if capture {n*PREPARED} else {0} || !out.iter().all(|v|v.is_finite()) {return Err("Delta projected output".into());}
    Ok((out,bytes))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oversized_capture_reply_rejects_before_weight_reads() {
        let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
        let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"delta_project_capture","tensor":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight","dims":[132,14,0,1],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap();
        let values=vec![0.;132*COLS+3*14*256+14*16384];
        assert_eq!(evaluate(&r,&values,&m,&mut |_,_|->Result<Vec<u8>> {panic!("oversize read")}).unwrap_err(),"Delta projected reply byte bound");
    }
    #[test]
    fn malformed_bounds_reject_without_reads() {
        let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
        for dims in [vec![],vec![0,16,0,0],vec![133,16,0,0],vec![1,17,0,0],vec![1,16,1,0],vec![1,16,18,0],vec![1,16,0,2],vec![usize::MAX,16,0,0]] {
            let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"delta_project_capture","tensor":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight","dims":dims,"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap();
            assert!(evaluate(&r,&[],&m,&mut |_,_|->Result<Vec<u8>> {panic!("invalid read")}).is_err());
        }
    }
}
