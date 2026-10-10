//! One query connects full Delta to the first MLP generation chunk.
use crate::{Manifest, Request, Result, prepared_weights::WeightBuffer};
pub(crate) const NAME: &str = "delta-mlp-start-exact-v1";
const C:usize=2560;
const CONV:usize=3*8192;
pub(crate) fn metadata(r:&Request)->Result<(usize,usize,usize,usize)> {
    if r.encoding!=NAME || !matches!(r.op.as_str(),"delta_mlp_stream_prepare"|"delta_mlp_stream_start_ids") || r.dims.len()!=3 || r.aux.len()!=1 || r.scalars.len()!=2 || r.scalars[0].to_bits()!=2f32.to_bits() || r.scalars[1].to_bits()!=1e-6f32.to_bits() {return Err("Delta MLP start metadata".into());}
    let(n,b,p)=(r.dims[0],r.dims[1],r.dims[2]);
    if !(1..=91).contains(&n) || b==0 || b>9216 || b%256!=0 || !(1..=132).contains(&p) {return Err("Delta MLP start bounds".into());}
    let s=r.tensor.strip_prefix("model.language_model.layers.").and_then(|s|s.strip_suffix(".post_attention_layernorm.weight")).ok_or("Delta MLP start layer")?;
    let layer:usize=s.parse().map_err(|_|"Delta MLP start layer")?;
    if (r.op=="delta_mlp_stream_start_ids" && layer!=0) || layer>30 || layer==30&&!cfg!(feature="experimental-terminal-stream") || layer.to_string()!=s || (layer+1)%4==0 || r.aux[0]!=format!("model.language_model.layers.{}.input_layernorm.weight",layer+1) {return Err("Delta MLP start scope".into());}
    Ok((n,b,p,layer))
}
fn mlp(r:&Request)->Result<Request> {
    let(n,b,_,_)=metadata(r)?;let mut m=r.clone();m.encoding=crate::mlp_stream::NAME.into();m.op="mlp_stream_prepare".into();m.dims=vec![n,0,b];Ok(m)
}
pub(crate) fn limit(r:&Request)->Result<usize> {Ok(crate::mlp_stream::limit(&mlp(r)?)?+CONV)}
/// Only the checked wire constructor can create this query operand.
pub struct PreparedDeltaMlpStart {request:Request,ids:Vec<f32>,hidden:Vec<f32>,norm:Vec<f32>,history:Vec<f32>,log:Vec<f32>}
impl PreparedDeltaMlpStart {
    pub(crate) fn decode(r:&Request,payload:&[u8])->Result<Self> {
        let(n,_,p,_)=metadata(r)?;
        if r.op=="delta_mlp_stream_start_ids" {
            let bf=CONV+p*2048;let end=1+4*n;
            if payload.first()!=Some(&2) || payload.len()!=end+2*bf+4*p*4128 {return Err("Delta MLP ids length/direction".into());}
            let mut ids=Vec::with_capacity(n);
            for b in payload[1..end].chunks_exact(4) {let id=u32::from_le_bytes(b.try_into().unwrap());if id>16_777_216 {return Err("Delta MLP ids range".into());}ids.push(id as f32);}
            let mut v=vec![0.;bf];crate::bf16_codec::unpack(&payload[end..end+2*bf],&mut v);let mut log=v[CONV..].to_vec();log.extend(payload[end+2*bf..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())));
            if !v.iter().chain(&log).all(|v|v.is_finite()) || !log[p*6144..].iter().all(|v|(0.0..=1.0).contains(v)) {return Err("Delta MLP ids finite/gates".into());}
            return Ok(Self{request:r.clone(),ids,hidden:vec![],norm:vec![],history:v[..CONV].to_vec(),log});
        }
        let bf=2*n*C+CONV+p*2048;let floats=p*4128;
        if payload.first()!=Some(&1) || payload.len()!=1+2*bf+4*floats {return Err("Delta MLP start input direction/length".into());}
        let mut values=vec![0.;bf];crate::bf16_codec::unpack(&payload[1..1+2*bf],&mut values);
        let mut log=values[2*n*C+CONV..].to_vec();log.extend(payload[1+2*bf..].chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())));
        if !values.iter().chain(&log).all(|v|v.is_finite()) || !log[p*6144..].iter().all(|v|(0.0..=1.0).contains(v)) {return Err("Delta MLP start input finite/gates".into());}
        Ok(Self{request:r.clone(),ids:vec![],hidden:values[..n*C].to_vec(),norm:values[n*C..2*n*C].to_vec(),history:values[2*n*C..2*n*C+CONV].to_vec(),log})
    }
    pub(crate) fn evaluate<F,B>(self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        self.evaluate_inner(r,m,read,false).and_then(|(reply,bytes)|Ok((reply.into_values()?,bytes)))
    }
    #[cfg(feature="experimental-direct-mlp-reply")]
    pub(crate) fn evaluate_reply<F,B>(self,r:&Request,m:&Manifest,read:&mut F)->Result<(crate::EvaluatedReply,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        self.evaluate_inner(r,m,read,true)
    }
    fn evaluate_inner<F,B>(self,r:&Request,m:&Manifest,read:&mut F,direct:bool)->Result<(crate::EvaluatedReply,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        if !crate::same_request(r,&self.request) {return Err("Delta MLP start identity".into());}
        let(n,_,p,layer)=metadata(r)?;
        let(hidden,norm,mut bytes)=if !self.ids.is_empty() {
            let mut er=r.clone();er.op="embed".into();er.encoding="bf16-block256-exact-v1".into();er.tensor="model.language_model.embed_tokens.weight".into();er.dims=vec![n,C];er.aux.clear();er.scalars.clear();
            let(hidden,mut bytes)=crate::evaluate_with_prepared_buffer(&er,&self.ids,m,&mut *read)?;
            let mut nr=er.clone();nr.op="rms_bf16".into();nr.tensor="model.language_model.layers.0.input_layernorm.weight".into();nr.scalars=vec![1e-6];
            let t=m.tensors.iter().find(|t|t.name==nr.tensor).ok_or("Delta MLP ids norm")?;
            if t.rows!=1 || t.cols!=C || t.dtype!="bf16" || t.bytes!=2*C as u64 {return Err("Delta MLP ids norm shape".into());}
            let(w,used)=crate::load_prepared_weight(t,&nr,&mut *read)?;bytes+=used;let norm=crate::execute(&nr,&hidden,&w)?;(hidden,norm,bytes)
        }else{(self.hidden,self.norm,0)};
        let mut dr=r.clone();dr.op="delta_project_capture".into();dr.encoding="bf16-block256-exact-v1".into();dr.tensor=format!("model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight");dr.dims=vec![n,32,0,0];dr.aux.clear();dr.scalars.clear();
        let(prep,used)=crate::delta_head_continue::Preparation::capture(&dr,&norm,m,read)?;bytes+=used;
        let(gated,history,used)=crate::profile::measure("start_all_delta_heads",||crate::delta_head_continue::group(&dr,&prep,&self.history,&self.log,n,32,0,p,m,read))?;bytes+=used;
        let root=dr.tensor.strip_suffix(".in_proj_qkv.weight").unwrap();
        let mut op=dr.clone();op.op="lora_integer".into();op.tensor=format!("{root}.out_proj.weight");op.dims=vec![n,C,4096,0];op.scalars=vec![2.];op.aux=vec![format!("{root}.out_proj.lora_A.weight"),format!("{root}.out_proj.lora_B.weight")];
        let(attention,used)=crate::profile::measure("start_delta_out",||crate::evaluate_integer_with_ax(&op,&gated,m,read,None,None))?;bytes+=used;
        let mr=mlp(r)?;let mut pair=hidden;pair.extend(attention);
        #[cfg(feature="experimental-direct-mlp-reply")]
        if direct {
            let(reply,used)=crate::profile::measure("start_mlp_front",||crate::mlp_stream::prepare_direct_reply(&mr,pair,m,read))?;
            let crate::EvaluatedReply::Payload(payload)=reply else {return Err("Delta MLP typed carry expected".into());};
            return Ok((wrap_reply(r,payload,&history)?,bytes+used));
        }
        #[cfg(not(feature="experimental-direct-mlp-reply"))]
        let _=direct;
        let(mut out,used)=crate::profile::measure("start_mlp_front",||crate::mlp_stream::prepare_direct(&mr,pair,m,read))?;bytes+=used;out.extend(history);Ok((crate::EvaluatedReply::Values(out),bytes))
    }
}
#[cfg(feature="experimental-direct-mlp-reply")]
fn wrap_reply(r:&Request,carry:crate::PayloadReply,history:&[f32])->Result<crate::EvaluatedReply> {
    metadata(r)?;
    if history.len()!=CONV || !crate::bf16_codec::classify_finite(history)? {return Err("Delta MLP typed history".into());}
    let inner=carry.into_payload();let mut payload=Vec::with_capacity(1+inner.len()+2*CONV);payload.push(0);payload.extend(inner);
    let start=payload.len();payload.resize(start+2*CONV,0);crate::bf16_codec::pack(history,&mut payload[start..]);
    Ok(crate::EvaluatedReply::payload(r,payload))
}
pub(crate) fn append(b:&mut Vec<u8>,r:&Request,x:&[f32])->Result<()> {
    if x.len()!=limit(r)? || !x.iter().all(|v|v.is_finite()) {return Err("Delta MLP start reply shape/finite".into());}
    let end=x.len()-CONV;b.push(0);crate::mlp_stream::append(b,&mlp(r)?,&x[..end])?;
    if !crate::bf16_codec::all_bf16(&x[end..]) {return Err("Delta MLP start history precision".into());}
    let start=b.len();b.resize(start+2*CONV,0);crate::bf16_codec::pack(&x[end..],&mut b[start..]);Ok(())
}
pub(crate) fn decode_reply(r:&Request,payload:&[u8])->Result<Vec<f32>> {
    metadata(r)?;if payload.first()!=Some(&0) || payload.len()<1+2*CONV {return Err("Delta MLP start reply direction".into());}
    let end=payload.len()-2*CONV;let mut values=crate::mlp_stream::decode_values(&mlp(r)?,&payload[1..end])?;
    if values.len()!=limit(r)?-CONV {return Err("Delta MLP start reply count".into());}
    let mut history=vec![0.;CONV];crate::bf16_codec::unpack(&payload[end..],&mut history);
    if !history.iter().all(|v|v.is_finite()) {return Err("Delta MLP start history finite".into());}
    values.extend(history);Ok(values)
}
#[cfg(test)]mod tests {
 use super::*;
 fn req()->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"delta_mlp_stream_prepare","encoding":NAME,"tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[1,4864,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"]})).unwrap()}
 #[test]fn terminal_layer_requires_explicit_feature_and_keeps_ids_layer_zero_only() {
  let mut r=req();r.tensor="model.language_model.layers.30.post_attention_layernorm.weight".into();r.aux=vec!["model.language_model.layers.31.input_layernorm.weight".into()];
  assert_eq!(metadata(&r).is_ok(),cfg!(feature="experimental-terminal-stream"));
  r.op="delta_mlp_stream_start_ids".into();assert!(metadata(&r).is_err());
  r.op="delta_mlp_stream_prepare".into();r.tensor="model.language_model.layers.31.post_attention_layernorm.weight".into();r.aux[0]="model.language_model.layers.32.input_layernorm.weight".into();assert!(metadata(&r).is_err());
 }
 #[test]fn ids_scope_range_and_direction_are_checked(){
  let mut r=req();r.op="delta_mlp_stream_start_ids".into();let mut raw=vec![2];raw.extend(1u32.to_le_bytes());raw.extend(vec![0;2*(CONV+45*2048)+4*45*4128]);
  assert!(PreparedDeltaMlpStart::decode(&r,&raw).is_ok());raw[1..5].copy_from_slice(&16_777_217u32.to_le_bytes());assert!(PreparedDeltaMlpStart::decode(&r,&raw).is_err());
  r.tensor=r.tensor.replace(".0.",".1.");r.aux[0]=r.aux[0].replace(".1.",".2.");assert!(metadata(&r).is_err());
 }
 #[test]fn bounds_identity_and_precision_reject_before_reads(){
  let r=req();let size=1+2*(2*C+CONV+45*2048)+4*45*4128;let mut raw=vec![0;size];raw[0]=1;
  let state=PreparedDeltaMlpStart::decode(&r,&raw).unwrap();let mut wrong=r.clone();wrong.dims[1]=5120;
  let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
  assert_eq!(state.evaluate(&wrong,&m,&mut|_,_|->Result<Vec<u8>>{panic!("identity read")}).unwrap_err(),"Delta MLP start identity");
  for dims in [vec![],vec![0,4864,45],vec![92,4864,45],vec![1,0,45],vec![1,4865,45],vec![1,4864,133],vec![usize::MAX,4864,45]] {let mut bad=r.clone();bad.dims=dims;assert!(metadata(&bad).is_err());}
  raw[size-4..].copy_from_slice(&1.1f32.to_le_bytes());assert!(PreparedDeltaMlpStart::decode(&r,&raw).is_err());raw[0]=0;assert!(PreparedDeltaMlpStart::decode(&r,&raw).is_err());
 }
}

#[cfg(all(test,feature="experimental-direct-mlp-reply"))]
mod direct_tests {
 use super::*;
 fn request(n:usize,b:usize)->Request {
  serde_json::from_value(serde_json::json!({"version":3,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":3,"op":"delta_mlp_stream_start_ids","encoding":NAME,"tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[n,b,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"]})).unwrap()
 }
 fn carry(r:&Request)->(Request,Vec<f32>,Vec<u8>) {
  let(n,b,_,_)=metadata(r).unwrap();let mr=mlp(r).unwrap();let mut v=vec![0.;crate::mlp_stream::limit(&mr).unwrap()];
  v[..n*C].fill(-0.);v[2*n*C..2*n*C+n*C/256].fill(0.0123456);
  let scales=2*n*C+n*C/256+2*n*64+n*b;v[scales..scales+n*(b/256)].fill(0.0234567);
  let mut payload=vec![];crate::mlp_stream::append(&mut payload,&mr,&v).unwrap();(mr,v,payload)
 }
 #[test]fn outer_typed_reply_preserves_entire_frame() {
  for n in [1,7,80,87,89,91] {for b in [4096,4352,9216] {
   let mut r=request(n,b);let(mr,mut values,payload)=carry(&r);let history=vec![-0.;CONV];values.extend(&history);
   for version in [1,2,3] {
    if version==2 && !cfg!(feature="experimental-blake3") {continue;}
    r.version=version;let mut inner=mr.clone();inner.version=version;
    let mut output=r.clone();output.step+=1;
    for signed in [false,true] {
     let crate::EvaluatedReply::Payload(p)=crate::EvaluatedReply::payload(&inner,payload.clone())else{unreachable!()};
     let actual=wrap_reply(&r,p,&history).unwrap().encode(&output,signed).unwrap();
     let expected=crate::encode_impl(&output,&values,signed && cfg!(feature="experimental-host-checksum")).unwrap();assert_eq!(actual,expected);
    }
   }
  }}
 }
 #[test]fn history_precision_and_identity_fail_before_reads() {
  let r=request(1,4352);let(mr,_,payload)=carry(&r);
  for history in [vec![0.;CONV-1],vec![f32::NAN;CONV],vec![0.1234567;CONV]] {
   let crate::EvaluatedReply::Payload(p)=crate::EvaluatedReply::payload(&mr,payload.clone())else{unreachable!()};assert!(wrap_reply(&r,p,&history).is_err());
  }
  let mut input=vec![2];input.extend(1u32.to_le_bytes());input.extend(vec![0;2*(CONV+45*2048)+4*45*4128]);
  let state=PreparedDeltaMlpStart::decode(&r,&input).unwrap();let mut changed=r.clone();changed.step+=1;
  let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
  assert_eq!(state.evaluate_reply(&changed,&m,&mut|_,_|->Result<Vec<u8>>{panic!("identity read")}).err().unwrap(),"Delta MLP start identity");
 }
}
