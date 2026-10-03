//! Keep layer30 MLP and layer31 terminal inference in one ordinary query.
//! All original BF16 boundaries remain; layer30 hidden and final KV are returned.
use crate::{Manifest,Request,Result,prepared_weights::WeightBuffer};
const C:usize=2560;
/// A decoded immutable tail input; only the checked decoder constructs it.
/// ```compile_fail
/// let mut input: imajev_runtime::PreparedTail = todo!();
/// input.values.clear();
/// ```
pub struct PreparedTail {request:Request,values:Vec<f32>}
fn shape(r:&Request)->Result<(usize,usize)> {
 if r.op!="terminal_tail_integer" || r.tensor!="model.language_model.layers.30.post_attention_layernorm.weight"
  || !matches!(r.encoding.as_str(),"bf16-exact"|"bf16-block256-exact-v1")
  || r.dims.len()!=2 || !(1..=89).contains(&r.dims[0]) || r.dims[1]>512 || r.dims[0]+r.dims[1]>512
  || !r.scalars.is_empty() || !r.aux.is_empty() {return Err("terminal tail metadata".into());}
 Ok((r.dims[0],r.dims[1]))
}
impl PreparedTail {
 pub(crate) fn decode(r:&Request,payload:&[u8])->Result<Self> {
  let(n,p)=shape(r)?;
  // decode_values verifies finiteness once, after the complete frame checksum.
  let values=crate::decode_values(r,payload)?;
  if values.len()!=2*n*C+p*2048 || values.len()>crate::MAX_FLOATS {return Err("terminal tail input".into());}
  Ok(Self{request:r.clone(),values})
 }
 pub(crate) fn evaluate<F,B>(&self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
 where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
  if !crate::same_request(r,&self.request) {return Err("terminal tail request identity".into());}
  core(r,&self.values,m,read)
 }
}
pub(crate) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 let(n,p)=shape(r)?;
 if x.len()!=2*n*C+p*2048 || x.len()>crate::MAX_FLOATS || !x.iter().all(|v|v.is_finite()) {return Err("terminal tail input".into());}
 core(r,x,m,read)
}
// Both entry points establish complete shape and finite values before this call.
fn core<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 let n=r.dims[0];let count=n*C;
 let mut mr=r.clone();mr.op="mlp_full_integer".into();mr.dims=vec![n,C];mr.scalars=vec![2.,1e-6];mr.aux=vec!["model.language_model.layers.31.input_layernorm.weight".into()];
 let(mut mlp,rm)=crate::mlp_pipeline::full(&mr,&x[..2*count],m,read)?;
 if mlp.len()!=2*count {return Err("terminal tail MLP output".into());}
 let mut tr=r.clone();tr.op="terminal_attention_mlp_integer".into();tr.tensor="model.language_model.layers.31.self_attn.q_proj.weight".into();
 // MLP execute checks finite outputs; decoded/validated history is immutable.
 // Pass slices directly: no second concatenation or finite rescan at the join.
 let(terminal,rt)=crate::terminal_attention::evaluate_validated_parts(&tr,&mlp[count..],&mlp[count-C..count],&x[2*count..],m,read)?;
 if terminal.len()!=2*C+n*2048 {return Err("terminal tail final output".into());}
 mlp.truncate(count);mlp.extend(terminal);
 Ok((mlp,rm.checked_add(rt).ok_or("terminal tail read overflow")?))
}
#[cfg(test)]mod tests {
 use super::*;
 #[test]fn typed_decoder_rejects_nonfinite_shape_and_identity_before_read(){
  let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"terminal_tail_integer","tensor":"model.language_model.layers.30.post_attention_layernorm.weight","dims":[1,0],"scalars":[],"encoding":"bf16-exact"})).unwrap();
  let frame=crate::encode(&r,&vec![0.;2*C]).unwrap();let(_,input)=crate::decode_query(&frame).unwrap();assert!(input.values().is_none());
  let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
  for field in 0..8 {let mut other=r.clone();match field {0=>other.step+=1,1=>other.input_hash="d".repeat(64),2=>other.tensor="wrong".into(),3=>other.dims[0]=2,4=>other.op="mlp_full_integer".into(),5=>other.aux.push("extra".into()),6=>other.scalars.push(0.),_=>other.encoding="bf16-block256-exact-v1".into()};assert!(crate::evaluate_decoded_with_prepared_buffer(&other,&input,&m,|_,_|->Result<Vec<u8>>{panic!("mismatched input read")}).is_err());}
  let mut wrong=r.clone();wrong.dims[0]=2;assert!(crate::decode_query(&crate::encode(&wrong,&vec![0.;2*C]).unwrap()).is_err());
  let mut invalid=frame;let header=u32::from_le_bytes(invalid[..4].try_into().unwrap())as usize;let pos=4+header+4+(2*C).div_ceil(8);invalid[pos..pos+2].copy_from_slice(&0x7fc0u16.to_le_bytes());invalid.truncate(invalid.len()-32);let hash=crate::frame_digest(1,&invalid,true).unwrap();invalid.extend(hash);assert!(crate::decode_query(&invalid).is_err());
 }
 #[test]fn reject_bad_contract_before_weight_reads(){
  let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
  let r:Request=serde_json::from_value(serde_json::json!({"version":2,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"terminal_tail_integer","tensor":"model.language_model.layers.30.post_attention_layernorm.weight","dims":[1,0],"scalars":[],"encoding":"bf16-exact"})).unwrap();
  for dims in [vec![],vec![0,0],vec![90,0],vec![1,512],vec![usize::MAX,0]] {let mut bad=r.clone();bad.dims=dims;assert!(evaluate(&bad,&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());}
  for field in 0..5 {let mut bad=r.clone();match field {0=>bad.op="mlp_full_integer".into(),1=>bad.tensor="model.language_model.layers.31.post_attention_layernorm.weight".into(),2=>bad.encoding="int8-block256-v1".into(),3=>bad.aux.push("extra".into()),_=>bad.scalars.push(2.)};assert!(evaluate(&bad,&vec![0.;2*C],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());}
  for x in [vec![0.;2*C-1],{let mut x=vec![0.;2*C];x[0]=f32::NAN;x}] {assert!(evaluate(&r,&x,&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());}
 }
}
