//! Finish the second Delta head group and consume its gated output locally.
use crate::{Manifest,Request,Result,MAX_FLOATS,prepared_weights::WeightBuffer};
const PREPARED:usize=2762;
pub(super) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    if r.op!="delta_project_finish" || !crate::lossless_encoding(&r.encoding) || r.dims.len()!=4
        || !r.aux.is_empty() || !r.scalars.is_empty() {
        return Err("Delta finish metadata".into());
    }
    let (n,h,first,keep)=(r.dims[0],r.dims[1],r.dims[2],r.dims[3]);
    if n==0 || n>90 || h!=16 || first!=16 || keep>1 {return Err("Delta finish shape".into());}
    let history=3*16*256;let state=16*16384;let half=n*16*128;
    let reuse_len=n*PREPARED+history+state;
    if x.len()!=reuse_len+half || x.len()>MAX_FLOATS || !crate::bf16_codec::all_bf16(&x[reuse_len..]) || !x[reuse_len..].iter().all(|v|v.is_finite()) {
        return Err("Delta finish input bounds".into());
    }
    let output=n*2560+history+state*keep;
    let bitmap=if r.encoding=="bf16-exact" {output.div_ceil(8)} else {output.div_ceil(256).div_ceil(8)};
    let header=serde_json::to_vec(r).map_err(|e|e.to_string())?.len()+1;
    if output>MAX_FLOATS || 4+header+32+4+bitmap+2*(n*2560+history)+4*state*keep>2_000_000 {
        return Err("Delta finish reply bounds".into());
    }
    let root=r.tensor.strip_suffix(".in_proj_qkv.weight").filter(|s|s.ends_with(".linear_attn")).ok_or("Delta finish tensor")?;
    let mut p=r.clone();p.op="lora_integer".into();p.tensor=format!("{root}.out_proj.weight");
    p.dims=vec![n,2560,4096,0];p.scalars=vec![2.];
    p.aux=vec![format!("{root}.out_proj.lora_A.weight"),format!("{root}.out_proj.lora_B.weight")];
    let base=m.tensors.iter().find(|t|t.name==p.tensor).ok_or("Delta finish base")?;
    let a=m.tensors.iter().find(|t|t.name==p.aux[0]).ok_or("Delta finish A")?;
    let b=m.tensors.iter().find(|t|t.name==p.aux[1]).ok_or("Delta finish B")?;
    if base.dtype!="int8" || base.rows!=2560 || base.cols!=4096 || a.dtype!="f32" || a.rows!=64 || a.cols!=4096
        || b.dtype!="f32" || b.rows!=2560 || b.cols!=64 {return Err("Delta finish weights".into());}
    // The existing reuse helper validates the prepared/history/state prefix.
    // Validate only the newly received first half above instead of scanning
    // that prefix twice. No weight reads occurred during manifest checks.
    let mut reuse=r.clone();reuse.op="delta_project_reuse".into();
    let (second,bytes)=crate::delta_projected::evaluate(&reuse,&x[..reuse_len],m,read)?;
    if second.len()!=half+history+state*keep {return Err("Delta finish reused output".into());}
    let mut gated=vec![0.;n*4096];
    for t in 0..n {
        gated[t*4096..t*4096+2048].copy_from_slice(&x[reuse_len+t*2048..reuse_len+(t+1)*2048]);
        gated[t*4096+2048..(t+1)*4096].copy_from_slice(&second[t*2048..(t+1)*2048]);
    }
    let (mut projected,used)=crate::evaluate_integer_with_ax(&p,&gated,m,read,None,None)?;
    projected.extend_from_slice(&second[half..]);
    // Both helpers reject nonfinite output; concatenation only copies values.
    if projected.len()!=output {return Err("Delta finish output".into());}
    Ok((projected,bytes+used))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bad_shape_and_first_half_reject_before_reads() {
        let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
        let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"delta_project_finish","tensor":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight","dims":[1,16,16,0],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap();
        for dims in [vec![],vec![0,16,16,0],vec![91,16,16,0],vec![1,14,16,0],vec![1,16,0,0],vec![1,16,16,2],vec![usize::MAX,16,16,0]] {
            r.dims=dims;assert!(evaluate(&r,&[],&m,&mut |_,_|->Result<Vec<u8>> {panic!("invalid read")}).is_err());
        }
        r.dims=vec![1,16,16,0];let mut x=vec![0.;PREPARED+3*16*256+16*16384+2048];
        let last=x.len()-1;
        for bad in [1.0000001,f32::INFINITY,f32::NAN] {
            x[last]=bad;
            assert_eq!(evaluate(&r,&x,&m,&mut |_,_|->Result<Vec<u8>> {panic!("invalid read")}).unwrap_err(),"Delta finish input bounds");
        }
    }
}
