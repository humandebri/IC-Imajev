//! Attention projection fusion with unchanged integer/LoRA, norm, RoPE and gate boundaries.
use crate::{Request, Manifest, Result, MAX_FLOATS, prepared_weights::WeightBuffer};
const COLS:usize=2560;
fn projection(r:&Request,m:&Manifest,name:String,n:usize,rows:usize,start:usize)->Result<Request> {
    let prefix=name.strip_suffix(".weight").ok_or("attention projection name")?;
    let mut p=r.clone();p.op="lora_integer".into();p.tensor=name.clone();p.dims=vec![n,rows,COLS,start];p.scalars=vec![2.];
    p.aux=vec![format!("{prefix}.lora_A.weight"),format!("{prefix}.lora_B.weight")];
    let base=m.tensors.iter().find(|t|t.name==name).ok_or("attention base")?;
    let a=m.tensors.iter().find(|t|t.name==p.aux[0]).ok_or("attention A")?;
    let b=m.tensors.iter().find(|t|t.name==p.aux[1]).ok_or("attention B")?;
    let total=if name.ends_with(".q_proj.weight") {8192} else {1024};
    if base.rows!=total || base.cols!=COLS || base.dtype!="int8" || a.rows!=64 || a.cols!=COLS || a.dtype!="f32" || b.rows!=total || b.cols!=64 || b.dtype!="f32" || start.checked_add(rows).is_none_or(|end|end>total) {return Err("attention projection weights".into());}
    let work=n as u64*(rows as u64*COLS as u64+64*(rows+COLS) as u64);
    if n*rows>MAX_FLOATS || rows*COLS>30_000_000 || work>2_500_000_000 {return Err("attention projection work".into());}
    Ok(p)
}
fn head_major(x:&[f32],n:usize,heads:usize,width:usize)->Vec<f32> {
    let mut out=Vec::with_capacity(x.len());
    for h in 0..heads {for t in 0..n {out.extend_from_slice(&x[(t*heads+h)*width..(t*heads+h+1)*width]);}}
    out
}
fn norm_rope<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F,n:usize,heads:usize,offset:usize,name:String)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    let mut nr=r.clone();nr.op="norm_rope_heads_bf16".into();nr.tensor=name;nr.dims=vec![n,256,64,offset,heads];nr.scalars=vec![1e-6,10000000.];nr.aux.clear();
    crate::rope::evaluate(&nr,&head_major(x,n,heads,256),m,read)
}
pub(super) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    evaluate_shared(r,x,m,read,None)
}
pub(crate) fn evaluate_shared<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F,prepared:Option<&crate::int8_kernel::QuantizedRows>)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    if !crate::lossless_encoding(&r.encoding) || !r.aux.is_empty() || !r.scalars.is_empty() || !x.iter().all(|v|v.is_finite()) {return Err("attention fusion metadata".into());}
    if r.op=="attention_kv_integer" {
        if r.dims.len()!=2 {return Err("attention KV dims".into());}
        let (n,offset)=(r.dims[0],r.dims[1]);
        if n==0 || n>132 || offset>512 || n+offset>512 || x.len()!=n*COLS {return Err("attention KV bounds".into());}
        let prefix=r.tensor.strip_suffix(".self_attn.k_proj.weight").ok_or("attention KV tensor")?;
        let root=format!("{prefix}.self_attn");
        let k=projection(r,m,r.tensor.clone(),n,1024,0)?;
        let v=projection(r,m,format!("{root}.v_proj.weight"),n,1024,0)?;
        // Both independent projections use the exact same validated quantization.
        let owned;
        let q=if let Some(q)=prepared {if q.rows()!=n || q.cols()!=COLS{return Err("attention shared quantization shape".into());}q}else{owned=crate::profile::measure("activation_quantize",||crate::int8_kernel::quantize_rows(x,n,COLS))?;&owned};
        let (keys,kread)=crate::evaluate_integer(&k,x,m,read,Some(q))?;
        let (values,vread)=crate::evaluate_integer(&v,x,m,read,Some(q))?;
        let (rotated,nread)=norm_rope(r,&keys,m,read,n,4,offset,format!("{root}.k_norm.weight"))?;
        let mut out=Vec::with_capacity(n*2048);
        for t in 0..n {for h in 0..4 {out.extend_from_slice(&rotated[(h*n+t)*256..(h*n+t+1)*256]);}}
        out.extend(values);return Ok((out,kread+vread+nread));
    }
    if r.op!="attention_q_gqa_integer" || r.dims.len()!=5 {return Err("attention Q dims".into());}
    let (n,total,offset,first,heads)=(r.dims[0],r.dims[1],r.dims[2],r.dims[3],r.dims[4]);
    if n==0 || n>132 || total<n || total>512 || offset!=total-n || first%4!=0 || heads==0 || heads>16 || heads%4!=0 || first.checked_add(heads).is_none_or(|end|end>16) {return Err("attention Q bounds".into());}
    if prepared.is_some_and(|q|q.rows()!=n || q.cols()!=COLS){return Err("attention shared quantization shape".into());}
    let groups=heads/4;let count=n*COLS;let kv=groups*total*256;
    if x.len()!=count+2*kv || x.len()>MAX_FLOATS || heads*(n*(n+1)/2+n*offset)*256>75_000_000 {return Err("attention Q input/work".into());}
    let root=r.tensor.strip_suffix(".q_proj.weight").filter(|name|name.ends_with(".self_attn")).ok_or("attention Q tensor")?;
    let request=projection(r,m,r.tensor.clone(),n,heads*512,first*512)?;
    let (projected,pread)=crate::evaluate_integer(&request,&x[..count],m,read,prepared)?;
    let mut queries=Vec::with_capacity(n*heads*256);let mut gate=Vec::with_capacity(n*heads*256);
    for row in projected.chunks_exact(512) {queries.extend_from_slice(&row[..256]);gate.extend_from_slice(&row[256..]);}
    let (rotated,nread)=norm_rope(r,&queries,m,read,n,heads,offset,format!("{root}.q_norm.weight"))?;
    let mut input=rotated;input.extend_from_slice(&x[count..]);
    let mut ar=r.clone();ar.op="gqa_suffix_bf16".into();ar.dims=vec![n,256,heads,offset];
    let attended=crate::execute(&ar,&input,&[])?;
    let mut token_major=Vec::with_capacity(n*heads*256*2);
    for t in 0..n {for h in 0..heads {token_major.extend_from_slice(&attended[(h*n+t)*256..(h*n+t+1)*256]);}}
    token_major.extend(gate);ar.op="attention_gate".into();ar.dims=vec![n*heads*256];
    let result=crate::execute(&ar,&token_major,&[])?;
    Ok((result,pread+nread))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request(op:&str,dims:Vec<usize>)->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":op,"tensor":"model.language_model.layers.3.self_attn.q_proj.weight","dims":dims,"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap()}
    #[test]
    fn reject_shapes_and_missing_weights_before_any_read() {
        let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
        let mut read=|_,_|->Result<Vec<u8>> {panic!("invalid attention read")};
        for dims in [vec![],vec![0,1,1,0,4],vec![1,1,0,1,4],vec![1,1,0,0,3],vec![1,512,0,0,16],vec![133,133,0,0,4],vec![1,1,0,16,4],vec![usize::MAX,1,0,0,4]] {assert!(evaluate(&request("attention_q_gqa_integer",dims),&[],&m,&mut read).is_err());}
        assert!(evaluate(&request("attention_q_gqa_integer",vec![1,1,0,0,4]),&vec![0.;COLS+512],&m,&mut read).is_err());
        for dims in [vec![],vec![0,0],vec![133,0],vec![1,usize::MAX]] {assert!(evaluate(&request("attention_kv_integer",dims),&[],&m,&mut read).is_err());}
    }
    #[test]
    fn head_order_is_token_and_group_preserving() {assert_eq!(head_major(&[0.,1.,2.,3.,4.,5.,6.,7.],2,2,2),[0.,1.,4.,5.,2.,3.,6.,7.]);}
}
