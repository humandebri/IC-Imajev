//! Lossless client-held MLP carry and exact prefix log; no persistent query state.
use crate::{Manifest,Request,Result,prepared_weights::WeightBuffer};
pub(crate) const NAME:&str="mlp-delta-log-carry-exact-v1";
pub(crate) const HUFFMAN_NAME:&str="mlp-delta-huffman-carry-exact-v1";
pub(crate) fn is_encoding(s:&str)->bool{s==NAME || s==HUFFMAN_NAME}
const C:usize=2560;const H:usize=9216;const HISTORY:usize=24576;
fn metadata(r:&Request)->Result<(usize,usize,usize,bool)> {
 let front=r.op=="mlp_finish_delta_log_mlp_front";
 let fused=r.op=="mlp_finish_delta_log_integer" || front;
 let dims_ok=if front {cfg!(feature="experimental-mlp-stream") && r.dims.len()==4 && crate::mlp_pipeline::valid_partial_rows(r.dims[2]) && r.dims[3]>0 && r.dims[3]<9216 && r.dims[3]%256==0}else{r.dims.len()==2 || r.dims.len()==3 && crate::mlp_pipeline::valid_partial_rows(r.dims[2])};
 if !is_encoding(&r.encoding) || !(fused || r.op=="mlp_finish_partial_integer") || !dims_ok || !(1..=89).contains(&r.dims[0]) || r.dims[1]>132 || (fused && r.dims[1]==0) || (!fused && r.dims[1]!=0) || r.scalars.len()!=2 || r.scalars[0].to_bits()!=2f32.to_bits() || r.scalars[1].to_bits()!=1e-6f32.to_bits() || r.aux.len()!=1 {return Err("MLP Delta carry metadata".into());}
 if fused && r.dims[0]*C+HISTORY+r.dims[1]*6176>crate::MAX_FLOATS{return Err("carry Delta input bounds".into());}
 let layer=r.tensor.strip_prefix("model.language_model.layers.").and_then(|s|s.strip_suffix(".post_attention_layernorm.weight")).ok_or("MLP Delta carry tensor")?;
 let i:usize=layer.parse().map_err(|_|"MLP Delta carry layer")?;
 if i>=31 || i.to_string()!=layer || (fused && (i+2)%4==0) || r.aux[0]!=format!("model.language_model.layers.{}.input_layernorm.weight",i+1) {return Err("MLP Delta carry next layer".into());}
 Ok((r.dims[0],r.dims[1],i,fused))
}
fn inflate_exact(input:&[u8],count:usize)->Result<Vec<u8>> {
 if count==0 {return if input.is_empty(){Ok(vec![])}else{Err("carry empty stream".into())};}
 use miniz_oxide::inflate::{core::{decompress,DecompressorOxide,inflate_flags::*},TINFLStatus};
 let mut out=vec![0;count];
 let(status,read,written)=decompress(&mut DecompressorOxide::new(),input,&mut out,0,TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF|TINFL_FLAG_PARSE_ZLIB_HEADER);
 if status!=TINFLStatus::Done || read!=input.len() || written!=count {return Err("carry deflate length/trailing/checksum".into());}
 Ok(out)
}
fn interleave(planar:&[u8],width:usize)->Vec<u8>{let n=planar.len()/width;let mut out=vec![0;planar.len()];for i in 0..n {for b in 0..width {out[i*width+b]=planar[b*n+i];}}out}
pub struct PreparedCarry {request:Request,mlp_request:Request,mlp:crate::PreparedMlp,conv:Vec<f32>,log:Vec<f32>,layer:usize,fused:bool,rows:usize}
impl PreparedCarry {
 pub(crate) fn decode(r:&Request,payload:&[u8])->Result<Self>{
  let(n,p,layer,fused)=metadata(r)?;
  if payload.first()!=Some(&1) || (r.encoding==NAME && payload.len()<25) {return Err("carry request direction".into());}
  let expected=[n*C*2,n*H,n*100*4,if fused{HISTORY*2}else{0},p*2048*2,p*(4096+32)*4];
  let decoded=if r.encoding==HUFFMAN_NAME {
   crate::profile::measure("carry_planes",||crate::carry_planes::decode(&payload[1..],&[(n*C,2),(n*H,1),(n*100,4),(if fused{HISTORY}else{0},2),(p*2048,2),(p*4128,4)]))?
  }else{let mut cursor:usize=25;let mut decoded=Vec::with_capacity(6);
  for (i,count)in expected.into_iter().enumerate(){let len=u32::from_le_bytes(payload[1+i*4..5+i*4].try_into().unwrap())as usize;let end=cursor.checked_add(len).ok_or("carry stream range")?;let src=payload.get(cursor..end).ok_or("carry truncated")?;decoded.push(crate::profile::measure("carry_inflate",||inflate_exact(src,count))?);cursor=end;}
  if cursor!=payload.len(){return Err("carry trailing streams".into());}decoded};
  let mut state=vec![1];state.extend(interleave(&decoded[0],2));state.extend_from_slice(&decoded[1]);state.extend(interleave(&decoded[2],4));
  let mut inner=r.clone();inner.op="mlp_down_norm_prepared".into();inner.encoding=crate::mlp_pipeline::NAME.into();inner.dims=vec![n,C];
  let mlp=crate::PreparedMlp::decode(&inner,&state)?;
  let mut conv=vec![0.;if fused{HISTORY}else{0}];crate::bf16_codec::unpack(&interleave(&decoded[3],2),&mut conv);
  let mut log=vec![0.;p*2048];crate::bf16_codec::unpack(&interleave(&decoded[4],2),&mut log);
  let rest=interleave(&decoded[5],4);log.extend(rest.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())));
  if !conv.iter().chain(&log).all(|v|v.is_finite()) || !log[p*6144..].iter().all(|v|(0.0..=1.0).contains(v)){return Err("carry prefix finite/gates".into());}
  Ok(Self{request:r.clone(),mlp_request:inner,mlp,conv,log,layer,fused,rows:*r.dims.get(2).unwrap_or(&256)})
 }
 pub(crate) fn evaluate<F,B>(&self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
 where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
  if !crate::same_request(r,&self.request){return Err("carry request identity".into());}
  let(mut both,mut bytes)=crate::profile::measure("carry_mlp_finish",||self.mlp.evaluate_partial(&self.mlp_request,m,read,self.rows))?;
  if !self.fused {return Ok((both,bytes));}
  let n=r.dims[0];let count=n*C;let mut input=both[count..].to_vec();input.extend_from_slice(&self.conv);input.extend_from_slice(&self.log);
  let mut dr=r.clone();dr.op="delta_full_log_integer".into();dr.encoding="bf16-block256-exact-v1".into();dr.tensor=format!("model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",self.layer+1);dr.dims=vec![n,32,r.dims[1],0];dr.scalars.clear();dr.aux.clear();
  let(delta,used)=crate::profile::measure("carry_next_delta",||crate::delta_full_log::evaluate(&dr,&input,m,read))?;bytes+=used;
  #[cfg(feature="experimental-mlp-stream")]
  if r.op=="mlp_finish_delta_log_mlp_front" {
   let mut pair=both[..count].to_vec();pair.extend_from_slice(&delta[..count]);
   let(mut carry,used)=crate::profile::measure("carry_next_mlp_front",||crate::mlp_stream::prepare_direct(&front_mlp(r)?,pair,m,read))?;bytes+=used;
   both.truncate(count);both.append(&mut carry);both.extend_from_slice(&delta[count..]);return Ok((both,bytes));
  }
  both.truncate(count);both.extend(delta);Ok((both,bytes))
 }
}
#[cfg(feature="experimental-mlp-stream")]
fn front_mlp(r:&Request)->Result<Request>{let(n,_,layer,_)=metadata(r)?;let mut mr=r.clone();mr.encoding=crate::mlp_stream::NAME.into();mr.op="mlp_stream_prepare".into();mr.tensor=format!("model.language_model.layers.{}.post_attention_layernorm.weight",layer+1);mr.aux=vec![format!("model.language_model.layers.{}.input_layernorm.weight",layer+2)];mr.dims=vec![n,0,r.dims[3]];Ok(mr)}
pub(crate) fn reply_count(r:&Request)->Result<usize>{let(n,_,_,fused)=metadata(r)?;
 #[cfg(feature="experimental-mlp-stream")]
 if r.op=="mlp_finish_delta_log_mlp_front"{return Ok(n*C+crate::mlp_stream::limit(&front_mlp(r)?)?+HISTORY);}
 Ok(2*n*C+if fused{HISTORY}else{0})}
pub(crate) fn append_reply(b:&mut Vec<u8>,r:&Request,x:&[f32])->Result<()>{let(n,_,_,_)=metadata(r)?;if x.len()!=reply_count(r)?{return Err("carry reply shape".into());}
 #[cfg(feature="experimental-mlp-stream")]
 if r.op=="mlp_finish_delta_log_mlp_front"{let end=x.len()-HISTORY;if !crate::bf16_codec::classify_finite(&x[..n*C])?||!crate::bf16_codec::classify_finite(&x[end..])?{return Err("carry front BF16".into());}b.push(2);let start=b.len();b.resize(start+2*n*C,0);crate::bf16_codec::pack(&x[..n*C],&mut b[start..]);crate::mlp_stream::append(b,&front_mlp(r)?,&x[n*C..end])?;let start=b.len();b.resize(start+2*HISTORY,0);crate::bf16_codec::pack(&x[end..],&mut b[start..]);return Ok(());}
 b.push(0);crate::block_codec::append(b,x)}
pub(crate) fn decode_reply(r:&Request,p:&[u8])->Result<Vec<f32>>{let(n,_,_,_)=metadata(r)?;
 #[cfg(feature="experimental-mlp-stream")]
 if r.op=="mlp_finish_delta_log_mlp_front"{if p.first()!=Some(&2)||p.len()<1+2*(n*C+HISTORY){return Err("carry front direction/length".into());}let end=p.len()-2*HISTORY;let mut v=vec![0.;n*C];crate::bf16_codec::unpack(&p[1..1+2*n*C],&mut v);v.extend(crate::mlp_stream::decode_values(&front_mlp(r)?,&p[1+2*n*C..end])?);let mut history=vec![0.;HISTORY];crate::bf16_codec::unpack(&p[end..],&mut history);v.extend(history);if v.len()!=reply_count(r)? || !v.iter().all(|v|v.is_finite()){return Err("carry front finite/count".into());}return Ok(v);}
 if p.first()!=Some(&0){return Err("carry reply direction".into());}let v=crate::block_codec::decode(&p[1..])?;if v.len()!=reply_count(r)?{return Err("carry reply shape".into());}Ok(v)}
#[cfg(test)]mod tests {use super::*;
 #[test]fn inflate_rejects_trailing_truncated_and_wrong_lengths(){let raw=b"bounded carry";let packed=miniz_oxide::deflate::compress_to_vec_zlib(raw,1);assert_eq!(inflate_exact(&packed,raw.len()).unwrap(),raw);for n in [raw.len()-1,raw.len()+1]{assert!(inflate_exact(&packed,n).is_err());}let mut extra=packed.clone();extra.push(0);assert!(inflate_exact(&extra,raw.len()).is_err());assert!(inflate_exact(&packed[..packed.len()-1],raw.len()).is_err());assert!(inflate_exact(&[],1).is_err());assert!(inflate_exact(&packed,0).is_err());}
 #[test]fn planar_bits_round_trip(){let raw:Vec<_>=(0..104).map(|i|i as u8).collect();for width in [2,4]{let n=raw.len()/width;let planar:Vec<_>=(0..width).flat_map(|b|(0..n).map(move|i|(i*width+b)as u8)).collect();assert_eq!(interleave(&planar,width),raw);}}
 #[test]#[ignore="requires recorded canister carry fixtures; see CARRY_FIXTURE_ROOT"]
 fn python_huffman_matches_recorded_deflate_bytes(){
  let root=std::path::PathBuf::from(std::env::var("CARRY_FIXTURE_ROOT").expect("fixture directory"));
  for layer in [0,1,3] {
   let new=std::fs::read(root.join(format!("carry-huffman-{layer:02}.request.bin"))).unwrap();
   let old=std::fs::read(root.join(format!("mlp-delta-fusion-check-v2/{layer:02}-fused.request.bin"))).unwrap();
   fn payload(b:&[u8])->&[u8]{let h=u32::from_le_bytes(b[..4].try_into().unwrap())as usize;&b[4+h..b.len()-32]}
   let a=payload(&new);let b=payload(&old);let shapes=[(87*C,2),(87*H,1),(87*100,4),(HISTORY,2),(45*2048,2),(45*4128,4)];
   let groups=crate::carry_planes::decode(&a[1..],&shapes).unwrap();let mut cursor=25;
   for(i,&(count,width))in shapes.iter().enumerate(){let len=u32::from_le_bytes(b[1+i*4..5+i*4].try_into().unwrap())as usize;let raw=inflate_exact(&b[cursor..cursor+len],count*width).unwrap();assert_eq!(groups[i],raw,"layer {layer} group {i}");cursor+=len;}
   assert_eq!(cursor,b.len());assert!(crate::decode_query(&new).is_ok());
  }
 }

}
#[cfg(all(test,feature="experimental-mlp-stream"))]
mod front_tests {
 use super::*;
 #[test]fn front_reply_keeps_exact_parts_and_bounds(){
  let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_finish_delta_log_mlp_front","encoding":HUFFMAN_NAME,"tensor":"model.language_model.layers.1.post_attention_layernorm.weight","dims":[1,45,768,1792],"scalars":[2.,1e-6],"aux":["model.language_model.layers.2.input_layernorm.weight"]})).unwrap();
  let count=crate::mlp_stream::limit(&front_mlp(&r).unwrap()).unwrap();let mut v=vec![-0.;C];v.extend(vec![1.;count]);v.extend(vec![-0.;HISTORY]);let frame=crate::encode(&r,&v).unwrap();let(_,decoded)=crate::decode(&frame).unwrap();assert_eq!(v.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),decoded.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
  for dims in [vec![],vec![1,45,768],vec![1,45,768,0],vec![1,45,768,1793],vec![1,45,768,9216],vec![90,45,768,1792],vec![usize::MAX,45,768,1792]]{let mut bad=r.clone();bad.dims=dims;assert!(metadata(&bad).is_err());}
 }
}
