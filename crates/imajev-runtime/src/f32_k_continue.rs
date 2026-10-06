//! Exact column-order F32 continuation; accumulators remain client-held.
use crate::{Manifest,Request,Result,prepared_weights::{WeightBuffer,LoadedWeight}};
fn shape(r:&Request)->Result<(usize,usize,usize,usize,usize)> {
 if r.op!="matmul_k_continue" || r.dims.len()!=5 || !r.aux.is_empty() || !r.scalars.is_empty()
  || !r.tensor.ends_with(".lora_A.weight") || !matches!(r.encoding.as_str(),"bf16-block256-exact-v1"|"bf16-exact") {return Err("F32 continuation metadata".into());}
 let(n,rows,cols,begin,count)=(r.dims[0],r.dims[1],r.dims[2],r.dims[3],r.dims[4]);
 if n==0 || n>132 || rows==0 || rows>256 || cols==0 || cols>262144 || cols%64!=0 || count==0 || count%64!=0 || begin%64!=0
  || begin.checked_add(count).is_none_or(|v|v>cols) || rows.checked_mul(cols).is_none_or(|v|v>30_000_000)
  || count.checked_add(rows).and_then(|v|n.checked_mul(v)).is_none_or(|v|v>crate::MAX_FLOATS) {return Err("F32 continuation bounds".into());}
 Ok((n,rows,cols,begin,count))
}
/// Shape/finite validation stays attached to the request until execution.
/// ```compile_fail
/// let mut input: imajev_runtime::PreparedF32K = todo!();
/// input.values.clear();
/// ```
pub struct PreparedF32K {request:Request,values:Vec<f32>}
impl PreparedF32K {
 pub(crate) fn decode(r:&Request,payload:&[u8])->Result<Self> {
  let(n,rows,_,begin,count)=shape(r)?;let values=crate::decode_values(r,payload)?;
  if values.len()!=n*(count+rows) || (begin==0 && values[n*count..].iter().any(|v|v.to_bits()!=0)) {return Err("F32 continuation initial state/length".into());}
  Ok(Self{request:r.clone(),values})
 }
 pub(crate) fn evaluate<F,B>(&self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
 where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
  if !crate::same_request(r,&self.request){return Err("F32 continuation identity".into());}
  core(r,&self.values,m,read)
 }
}
pub(crate) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 let(n,rows,_,begin,count)=shape(r)?;
 if x.len()!=n*(count+rows) || !x.iter().all(|v|v.is_finite()) || (begin==0 && x[n*count..].iter().any(|v|v.to_bits()!=0)){return Err("F32 continuation input".into());}
 core(r,x,m,read)
}
fn core<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 let(n,rows,cols,begin,count)=shape(r)?;
 let t=m.tensors.iter().find(|t|t.name==r.tensor).ok_or("F32 continuation tensor")?;
 if t.dtype!="f32" || t.rows!=rows || t.cols!=cols || t.bytes!=(rows*cols*4)as u64{return Err("F32 continuation tensor shape".into());}
 let mut wr=r.clone();wr.op="matmul".into();wr.dims=vec![n,rows,cols];
 let(w,bytes)=crate::load_prepared_weight(t,&wr,&mut *read)?;
 let LoadedWeight::Prepared(w)=w else{return Err("F32 continuation requires fixed prepared output layout".into());};
 let out=crate::profile::measure("f32_k_continue",||crate::f32_output::continue_columns(&x[..n*count],&x[n*count..],&w,n,rows,cols,begin,count))?;
 Ok((out,bytes))
}
#[cfg(test)]mod tests {
 use super::*;
 fn req()->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"matmul_k_continue","tensor":"model.language_model.layers.0.mlp.down_proj.lora_A.weight","dims":[1,64,256,0,64],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap()}
 #[test]fn wrong_progress_identity_and_initial_state_reject_before_read(){
  let r=req();let frame=crate::encode(&r,&vec![0.;128]).unwrap();let(_,input)=crate::decode_query(&frame).unwrap();let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
  for field in 0..8 {let mut bad=r.clone();match field {0=>bad.step+=1,1=>bad.input_hash="d".repeat(64),2=>bad.tensor="wrong.lora_A.weight".into(),3=>bad.dims[3]=64,4=>bad.op="matmul".into(),5=>bad.aux.push("x".into()),6=>bad.scalars.push(2.),_=>bad.encoding="bf16-exact".into()};assert!(crate::evaluate_decoded_with_prepared_buffer(&bad,&input,&m,|_,_|->Result<Vec<u8>>{panic!("wrong identity read")}).is_err());}
  for value in [-0.,1.,f32::NAN] {let mut x=vec![0.;128];x[64]=value;assert!(evaluate(&r,&x,&m,&mut|_,_|->Result<Vec<u8>>{panic!("bad initial accumulator read")}).is_err());}
 }
 #[test]fn bounds_overflow_and_tensor_mismatch_reject_before_read(){
  let r=req();let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
  for dims in [vec![],vec![0,64,256,0,64],vec![133,64,256,0,64],vec![1,257,256,0,64],vec![1,64,256,1,64],vec![1,64,256,192,128],vec![1,64,256,usize::MAX,64],vec![1,64,256,0,0]]{let mut bad=r.clone();bad.dims=dims;assert!(evaluate(&bad,&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid shape read")}).is_err());}
  assert!(evaluate(&r,&vec![0.;128],&m,&mut|_,_|->Result<Vec<u8>>{panic!("missing tensor read")}).is_err());
 }
}
