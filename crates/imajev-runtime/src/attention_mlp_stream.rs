//! Client-held bridge across MLP completion, full-attention KV/Q and next MLP.
use crate::{Manifest,Request,Result,prepared_weights::WeightBuffer,int8_kernel::QuantizedRows};
pub(crate) const NAME:&str="attention-mlp-stream-exact-v1";
const C:usize=2560;const KV:usize=2048;
fn metadata(r:&Request)->Result<(usize,usize,usize,usize)> {
 let complete=r.op=="mlp_complete_attention_kv";
 if r.encoding!=NAME || !(complete || r.op=="attention_finish_mlp_front") || r.dims.len()!=if complete{4}else{3} || r.aux.len()!=1 || r.scalars.len()!=2 || r.scalars[0].to_bits()!=2f32.to_bits() || r.scalars[1].to_bits()!=1e-6f32.to_bits(){return Err("attention MLP metadata".into());}
 let n=r.dims[0];let b=r.dims[1];let p=r.dims[if complete{3}else{2}];
 if !(1..=89).contains(&n) || b==0 || b>=9216 || b%256!=0 || p>132 || n+p>512 || complete && b.checked_add(r.dims[2])!=Some(9216){return Err("attention MLP bounds".into());}
 let s=r.tensor.strip_prefix("model.language_model.layers.").and_then(|s|s.strip_suffix(".post_attention_layernorm.weight")).ok_or("attention MLP tensor")?;
 let layer:usize=s.parse().map_err(|_|"attention MLP layer")?;
 if layer.to_string()!=s || layer>=30 || layer%4!=if complete{2}else{3} || r.aux[0]!=format!("model.language_model.layers.{}.input_layernorm.weight",layer+1){return Err("attention MLP scope".into());}
 Ok((n,b,p,layer))
}
fn mlp(r:&Request)->Result<Request>{let(n,b,_,_)=metadata(r)?;let mut inner=r.clone();inner.encoding=crate::mlp_stream::NAME.into();inner.op=if r.op=="mlp_complete_attention_kv"{"mlp_stream_complete"}else{"mlp_stream_prepare"}.into();inner.dims=if r.op=="mlp_complete_attention_kv"{vec![n,b,9216-b]}else{vec![n,0,b]};Ok(inner)}
pub(crate) fn limit(r:&Request)->Result<usize>{let(n,_,_,_)=metadata(r)?;Ok(if r.op=="mlp_complete_attention_kv"{n*(3*C+KV+C/256)}else{crate::mlp_stream::limit(&mlp(r)?)?+n*KV})}
pub struct PreparedAttentionMlp {request:Request,stream:Option<crate::PreparedMlpStream>,values:Vec<f32>,quant:Option<QuantizedRows>}
impl PreparedAttentionMlp {
 pub(crate) fn decode(r:&Request,p:&[u8])->Result<Self>{
  let(n,_,offset,_)=metadata(r)?;
  if r.op=="mlp_complete_attention_kv"{return Ok(Self{request:r.clone(),stream:Some(crate::PreparedMlpStream::decode(&mlp(r)?,p)?),values:vec![],quant:None});}
  let bf=n*(2*C+KV)+offset*KV;let end=1+2*bf;let scales=n*C/256;
  if p.first()!=Some(&2) || p.len()!=end+n*C+4*scales{return Err("attention MLP carry length/direction".into());}
  let mut v=vec![0.;bf];crate::bf16_codec::unpack(&p[1..end],&mut v);
  if !v.iter().all(|v|v.is_finite()){return Err("attention MLP carry finite".into());}
  let sx:Vec<_>=p[end+n*C..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let q=QuantizedRows::from_bytes(n,C,&p[end..end+n*C],&sx)?;
  Ok(Self{request:r.clone(),stream:None,values:v,quant:Some(q)})
 }
 pub(crate) fn evaluate<F,B>(self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
  if !crate::same_request(r,&self.request){return Err("attention MLP identity".into());}
  let(n,_,p,layer)=metadata(r)?;let count=n*C;
  if let Some(stream)=self.stream {
   let(mut both,mut used)=crate::profile::measure("bridge_mlp_complete",||stream.evaluate(&mlp(r)?,m,read))?;
   if both.len()!=2*count{return Err("attention MLP complete shape".into());}
   let q=crate::profile::measure("bridge_attention_quantize_once",||crate::int8_kernel::quantize_rows(&both[count..],n,C))?;
   let mut kr=r.clone();kr.encoding="bf16-block256-exact-v1".into();kr.op="attention_kv_integer".into();kr.tensor=format!("model.language_model.layers.{}.self_attn.k_proj.weight",layer+1);kr.dims=vec![n,p];kr.scalars.clear();kr.aux.clear();
   let(kv,bytes)=crate::profile::measure("bridge_attention_kv",||crate::attention_fusion::evaluate_shared(&kr,&both[count..],m,read,Some(&q)))?;used+=bytes;both.extend(kv);q.append_wire_values(&mut both);both.extend_from_slice(&q.scales()[..n*C/256]);return Ok((both,used));
  }
  let v=self.values;let q=self.quant.ok_or("attention MLP quant state")?;let total=n+p;let kv=&v[2*count..2*count+n*KV];let prefix=&v[2*count+n*KV..];
  let mut payload=v[count..2*count].to_vec();
  for part in 0..2{for head in 0..4 {let start=part*p*1024+head*p*256;payload.extend_from_slice(&prefix[start..start+p*256]);for t in 0..n{let start=part*n*1024+(t*4+head)*256;payload.extend_from_slice(&kv[start..start+256]);}}}
  let root=format!("model.language_model.layers.{layer}.self_attn");let mut qr=r.clone();qr.op="attention_q_gqa_integer".into();qr.encoding="bf16-block256-exact-v1".into();qr.tensor=format!("{root}.q_proj.weight");qr.dims=vec![n,total,p,0,16];qr.scalars.clear();qr.aux.clear();
  let(gated,mut used)=crate::profile::measure("bridge_attention_q_gqa",||crate::attention_fusion::evaluate_shared(&qr,&payload,m,read,Some(&q)))?;
  let mut out=qr;out.op="lora_integer".into();out.tensor=format!("{root}.o_proj.weight");out.dims=vec![n,C,4096,0];out.scalars=vec![2.];out.aux=vec![format!("{root}.o_proj.lora_A.weight"),format!("{root}.o_proj.lora_B.weight")];
  let(attention,bytes)=crate::profile::measure("bridge_attention_out",||crate::evaluate_integer_with_ax(&out,&gated,m,read,None,None))?;used+=bytes;
  let mut pair=v[..count].to_vec();pair.extend(attention);let(mut state,bytes)=crate::profile::measure("bridge_mlp_front",||crate::mlp_stream::prepare_direct(&mlp(r)?,pair,m,read))?;used+=bytes;state.extend_from_slice(kv);Ok((state,used))
 }
}
pub(crate) fn append(b:&mut Vec<u8>,r:&Request,v:&[f32])->Result<()> {
 let(n,_,_,_)=metadata(r)?;if v.len()!=limit(r)? || !v.iter().all(|v|v.is_finite()){return Err("attention MLP reply count/finite".into());}
 if r.op=="mlp_complete_attention_kv"{let count=n*(2*C+KV);if !crate::bf16_codec::all_bf16(&v[..count]){return Err("attention MLP BF16 reply".into());}b.push(1);let pos=b.len();b.resize(pos+count*2,0);crate::bf16_codec::pack(&v[..count],&mut b[pos..]);let q=&v[count..count+n*C];if q.iter().any(|v|*v < -127. || *v>127. || *v!=v.trunc() || v.to_bits()==0x80000000) || v[count+n*C..].iter().any(|v|*v<=0.){return Err("attention MLP integer/scale reply".into());}b.extend(q.iter().map(|v|*v as i8 as u8));for v in &v[count+n*C..]{b.extend(v.to_le_bytes());}return Ok(());}
 b.push(3);let end=v.len()-n*KV;crate::mlp_stream::append(b,&mlp(r)?,&v[..end])?;if !crate::bf16_codec::all_bf16(&v[end..]){return Err("attention MLP KV precision".into());}let pos=b.len();b.resize(pos+2*n*KV,0);crate::bf16_codec::pack(&v[end..],&mut b[pos..]);Ok(())
}
pub(crate) fn decode_reply(r:&Request,p:&[u8])->Result<Vec<f32>> {
 let(n,_,_,_)=metadata(r)?;
 if r.op=="mlp_complete_attention_kv"{let count=n*(2*C+KV);let end=1+count*2;if p.first()!=Some(&1)||p.len()!=end+n*C+4*n*C/256{return Err("attention MLP reply length".into());}let mut v=vec![0.;count];crate::bf16_codec::unpack(&p[1..end],&mut v);v.extend(p[end..end+n*C].iter().map(|b|*b as i8 as f32));v.extend(p[end+n*C..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())));if !v.iter().all(|v|v.is_finite())||p[end..end+n*C].contains(&128)||v[count+n*C..].iter().any(|v|*v<=0.){return Err("attention MLP reply lanes".into());}return Ok(v);}
 if p.first()!=Some(&3)||p.len()<1+2*n*KV{return Err("attention MLP reply direction".into());}let end=p.len()-2*n*KV;let mut v=crate::mlp_stream::decode_values(&mlp(r)?,&p[1..end])?;let mut kv=vec![0.;n*KV];crate::bf16_codec::unpack(&p[end..],&mut kv);v.extend(kv);if v.len()!=limit(r)? || !v.iter().all(|v|v.is_finite()){return Err("attention MLP reply finite/count".into());}Ok(v)
}
#[cfg(test)]mod tests {
 use super::*;
 fn req()->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"attention_finish_mlp_front","encoding":NAME,"tensor":"model.language_model.layers.3.post_attention_layernorm.weight","dims":[1,4608,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.4.input_layernorm.weight"]})).unwrap()}
 #[test]fn checked_carry_identity_and_bounds(){
  let r=req();let count=2*C+KV+45*KV;let mut p=vec![2];p.extend(vec![0;2*count+C]);for _ in 0..C/256{p.extend(1f32.to_le_bytes());}
  let state=PreparedAttentionMlp::decode(&r,&p).unwrap();let mut changed=r.clone();changed.dims.clear();let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
  assert_eq!(state.evaluate(&changed,&m,&mut|_,_|->Result<Vec<u8>>{panic!("identity read")}).unwrap_err(),"attention MLP identity");
  for dims in [vec![],vec![0,4608,45],vec![90,4608,45],vec![1,9216,45],vec![1,4609,45],vec![1,4608,133],vec![usize::MAX,4608,45]]{let mut bad=r.clone();bad.dims=dims;assert!(metadata(&bad).is_err());}
  p.push(0);assert!(PreparedAttentionMlp::decode(&r,&p).is_err());p.pop();p[1+2*count]=128;assert!(PreparedAttentionMlp::decode(&r,&p).is_err());p[1+2*count]=0;p[1..3].copy_from_slice(&0x7fc0u16.to_le_bytes());assert!(PreparedAttentionMlp::decode(&r,&p).is_err());
 }
 #[test]fn reply_preserves_integer_scales_and_negative_zero(){let r=Request{op:"mlp_complete_attention_kv".into(),tensor:"model.language_model.layers.2.post_attention_layernorm.weight".into(),aux:vec!["model.language_model.layers.3.input_layernorm.weight".into()],dims:vec![1,4608,4608,45],..req()};let mut v=vec![-0.;2*C+KV];v.extend(vec![-127.;C]);v.extend(vec![0.0123;C/256]);let frame=crate::encode(&r,&v).unwrap();assert_eq!(crate::decode(&frame).unwrap().1.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),v.iter().map(|v|v.to_bits()).collect::<Vec<_>>());}
}
