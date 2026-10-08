//! Complete streamed MLP + next Attention + first chunk of the following MLP.
use super::*;
const C:usize=2560;
const KV:usize=2048;
#[derive(CandidType,Deserialize)]
pub(super) struct AttentionMlpFrontMeasurement {
 state:StateBytes,previous_hidden:StateBytes,kv:StateBytes,instructions:u64,
 stable_read_bytes:u64,heap_pages:u64,stable_pages:u64,spans:Vec<(String,u64)>,
}
fn bf16_bytes(x:&[f32])->Vec<u8>{x.iter().flat_map(|x|((x.to_bits()>>16)as u16).to_le_bytes()).collect()}
#[ic_cdk::query]
fn attention_mlp_front(state:StateBytes,front:u32)->Result<AttentionMlpFrontMeasurement,String>{
 query_access();let start=ic_cdk::api::performance_counter(0);
 if state.len()>1_990_000||front==0||front>=9216||front%256!=0{return Err("attention front input bounds".into());}
 let(r,input)=decode_query(&state)?;
 let text=r.tensor.strip_prefix("model.language_model.layers.").and_then(|s|s.strip_suffix(".post_attention_layernorm.weight")).ok_or("attention front layer")?;
 let layer=text.parse::<usize>().map_err(|_|"attention front layer")?;
 if layer>=30||layer%4!=2||text!=layer.to_string()||r.op!="mlp_stream_complete_attention_full"||r.encoding!="mlp-attention-finish-exact-v1"||r.dims.len()!=3||!(1..=69).contains(&r.dims[0]){return Err("attention front scope".into());}
 let n=r.dims[0];let before=ic_cdk::api::performance_counter(0);let(y,mut reads)=evaluate_decoded(&r,input)?;let values=y.into_values()?;
 if values.len()!=n*(2*C+KV){return Err("attention front result shape".into());}
 let mut spans=vec![("mlp_complete_attention_full".into(),ic_cdk::api::performance_counter(0)-before)];
 let mut mr=r.clone();mr.op="mlp_stream_prepare".into();mr.encoding="mlp-stream-exact-v1".into();mr.dims=vec![n,0,front as usize];
 mr.tensor=format!("model.language_model.layers.{}.post_attention_layernorm.weight",layer+1);mr.aux=vec![format!("model.language_model.layers.{}.input_layernorm.weight",layer+2)];
 let frame=encode(&mr,&values[..2*n*C])?;let(_,input)=decode_query(&frame)?;
 let before=ic_cdk::api::performance_counter(0);let(y,b)=evaluate_decoded(&mr,input)?;reads+=b;spans.push(("next_mlp_front".into(),ic_cdk::api::performance_counter(0)-before));
 mr.step=mr.step.checked_add(1).ok_or("attention front progress overflow")?;
 let state=y.encode(&mr,cfg!(feature="experimental-host-checksum"))?;let previous_hidden=bf16_bytes(&values[..n*C]);let kv=bf16_bytes(&values[2*n*C..]);
 if state.len()+previous_hidden.len()+kv.len()>1_990_000{return Err("attention front reply bounds".into());}
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Ok(AttentionMlpFrontMeasurement{state:state.into(),previous_hidden:previous_hidden.into(),kv:kv.into(),instructions:ic_cdk::api::performance_counter(0)-start,stable_read_bytes:reads,heap_pages,stable_pages:ic_cdk::api::stable_size(),spans})
}
