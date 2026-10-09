//! Complete attention in one ordinary query; prefix KV stays client-held.
use crate::{Request,Manifest,Result,prepared_weights::WeightBuffer};
const C:usize=2560;
pub(crate) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 if r.op!="attention_full_integer" || !crate::lossless_encoding(&r.encoding) || r.dims.len()!=3 || !r.aux.is_empty() || !r.scalars.is_empty(){return Err("full attention metadata".into());}
 let(n,offset,last)=(r.dims[0],r.dims[1],r.dims[2]);
 if n==0 || n>132 || offset>crate::MAX_SEQUENCE_TOKENS || n+offset>crate::MAX_SEQUENCE_TOKENS || last>1 || (n>91 && last==0) || x.len()>crate::MAX_FLOATS || x.len()!=n*C+offset*2048 || !x.iter().all(|v|v.is_finite()){return Err("full attention bounds".into());}
 let p=r.tensor.strip_suffix(".self_attn.q_proj.weight").ok_or("full attention tensor")?;
 let i=p.strip_prefix("model.language_model.layers.").ok_or("full attention layer")?;let layer:usize=i.parse().map_err(|_|"full attention layer")?;
 if layer>31 || layer%4!=3 || i!=layer.to_string() || (last==1 && layer!=31){return Err("full attention layer".into());}
 let root=format!("{p}.self_attn");let qn=if last==1 {1}else{n};
 // Validate every fixed weight before any weight read or projection allocation.
 for (name,rows,cols,dtype) in [
 ("k_proj.weight",1024,C,"int8"),("v_proj.weight",1024,C,"int8"),("q_proj.weight",8192,C,"int8"),("o_proj.weight",C,4096,"int8"),
 ("k_proj.lora_A.weight",64,C,"f32"),("v_proj.lora_A.weight",64,C,"f32"),("q_proj.lora_A.weight",64,C,"f32"),("o_proj.lora_A.weight",64,4096,"f32"),
 ("k_proj.lora_B.weight",1024,64,"f32"),("v_proj.lora_B.weight",1024,64,"f32"),("q_proj.lora_B.weight",8192,64,"f32"),("o_proj.lora_B.weight",C,64,"f32"),
 ("k_norm.weight",1,256,"bf16"),("q_norm.weight",1,256,"bf16")]{
  let name=format!("{root}.{name}");let t=m.tensors.iter().find(|t|t.name==name).ok_or("full attention missing weight")?;
  let bytes=if dtype=="f32"{rows*cols*4}else if dtype=="bf16"{rows*cols*2}else{rows*cols+rows*4};
  if t.rows!=rows || t.cols!=cols || t.dtype!=dtype || t.bytes!=bytes as u64 {return Err("full attention weight shape".into());}
 }
 let q=crate::int8_kernel::quantize_rows(&x[..n*C],n,C)?;
 let mut kr=r.clone();kr.op="attention_kv_integer".into();kr.tensor=format!("{root}.k_proj.weight");kr.dims=vec![n,offset];
 let(kv,rk)=crate::attention_fusion::evaluate_shared(&kr,&x[..n*C],m,read,Some(&q))?;
 if kv.len()!=n*2048 {return Err("full attention KV output".into());}
 let total=offset+n;let mut payload=Vec::with_capacity(qn*C+total*2048);payload.extend_from_slice(&x[(n-qn)*C..n*C]);
 for part in 0..2 {for h in 0..4 {
  let start=n*C+part*offset*1024+h*offset*256;payload.extend_from_slice(&x[start..start+offset*256]);
  for t in 0..n {let start=part*n*1024+(t*4+h)*256;payload.extend_from_slice(&kv[start..start+256]);}
 }}
 let mut qr=r.clone();qr.op="attention_q_gqa_integer".into();qr.dims=vec![qn,total,total-qn,0,16];
 let(gated,rq)=crate::attention_fusion::evaluate_shared(&qr,&payload,m,read,if qn==n{Some(&q)}else{None})?;
 let mut out=r.clone();out.op="lora_integer".into();out.tensor=format!("{root}.o_proj.weight");out.dims=vec![qn,C,4096,0];out.scalars=vec![2.];out.aux=vec![format!("{root}.o_proj.lora_A.weight"),format!("{root}.o_proj.lora_B.weight")];
 let(mut y,ro)=crate::evaluate_integer_with_ax(&out,&gated,m,read,None,None)?;y.extend(kv);
 if y.len()!=qn*C+n*2048 {return Err("full attention output".into());}Ok((y,rk+rq+ro))
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn bad_requests_fail_before_reads(){
 let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
 let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"attention_full_integer","tensor":"model.language_model.layers.3.self_attn.q_proj.weight","dims":[1,0,0],"scalars":[],"aux":[],"encoding":"bf16-block256-exact-v1"})).unwrap();
 for dims in [vec![],vec![0,0,0],vec![90,0,0],vec![1,512,0],vec![1,0,2],vec![usize::MAX,0,0],vec![1,0,1]]{r.dims=dims;assert!(evaluate(&r,&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());}
 r.dims=vec![1,0,0];assert!(evaluate(&r,&vec![0.;C],&m,&mut|_,_|->Result<Vec<u8>>{panic!("missing manifest read")}).is_err());
 }
}
