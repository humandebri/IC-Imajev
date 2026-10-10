//! Experimental exact MLP completion -> full Delta -> next MLP front query.
use super::*;
const C:usize=2560;
const H:usize=9216;
#[derive(CandidType,Deserialize)]
pub(super) struct MlpDeltaMeasurement {
    state:StateBytes, previous_hidden:StateBytes, conv:StateBytes,
    instructions:u64, stable_read_bytes:u64, heap_pages:u64, stable_pages:u64,
    spans:Vec<(String,u64)>,
}
fn bf16_bytes(v:&[f32])->Vec<u8> {v.iter().flat_map(|v|((v.to_bits()>>16)as u16).to_le_bytes()).collect()}
#[ic_cdk::query(name = "runDeltaInferenceStep")]
fn mlp_delta_front(state:StateBytes,prefix:StateBytes,p:u32,front:u32)->Result<MlpDeltaMeasurement,String> {
    query_access();let start=ic_cdk::api::performance_counter(0);
    if state.len()+prefix.len()>1_990_000 || !(1..=27).contains(&p) || front==0 || front>=H as u32 || front%256!=0 {return Err("bridge input bounds".into());}
    let(r,input)=decode_query(&state)?;
    let layer=r.tensor.strip_prefix("model.language_model.layers.").and_then(|s|s.strip_suffix(".post_attention_layernorm.weight")).and_then(|s|s.parse::<usize>().ok()).ok_or("bridge layer")?;
    if r.op!="mlp_stream_complete" || r.encoding!="mlp-stream-exact-v1" || r.dims.len()!=3 || !(1..=91).contains(&r.dims[0]) || layer>=30 || (layer+2)%4==0 {return Err("bridge scope".into());}
    let n=r.dims[0];let p=p as usize;let bf=24576+p*2048;
    if prefix.len()!=2*bf+4*p*4128 {return Err("bridge prefix shape".into());}
    let mut values=Vec::with_capacity(bf+p*4128);
    for v in prefix[..2*bf].chunks_exact(2){values.push(f32::from_bits((u16::from_le_bytes(v.try_into().unwrap())as u32)<<16));}
    for v in prefix[2*bf..].chunks_exact(4){values.push(f32::from_le_bytes(v.try_into().unwrap()));}
    if !values.iter().all(|v|v.is_finite()) || values[bf+p*4096..].iter().any(|v|!(0.0..=1.0).contains(v)) {return Err("bridge prefix finite/gates".into());}
    let before=ic_cdk::api::performance_counter(0);let(y,mut reads)=evaluate_decoded(&r,input)?;let both=y.into_values()?;
    if both.len()!=2*n*C {return Err("bridge MLP result shape".into());}
    let mut spans=vec![("mlp_complete".into(),ic_cdk::api::performance_counter(0)-before)];
    let mut dr=r.clone();dr.op="delta_full_log_integer".into();dr.encoding="bf16-block256-exact-v1".into();dr.tensor=format!("model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",layer+1);dr.dims=vec![n,32,p,0];dr.aux.clear();dr.scalars.clear();
    let mut x=both[n*C..].to_vec();x.extend_from_slice(&values[..24576]);x.extend_from_slice(&values[24576..]);
    let before=ic_cdk::api::performance_counter(0);let(out,b)=evaluate(&dr,&x)?;reads+=b;spans.push(("full_delta".into(),ic_cdk::api::performance_counter(0)-before));
    if out.len()!=n*C+24576 {return Err("bridge Delta result shape".into());}
    let mut mr=r.clone();mr.op="mlp_stream_prepare".into();mr.dims=vec![n,0,front as usize];mr.tensor=format!("model.language_model.layers.{}.post_attention_layernorm.weight",layer+1);mr.aux=vec![format!("model.language_model.layers.{}.input_layernorm.weight",layer+2)];
    let mut pair=both[..n*C].to_vec();pair.extend_from_slice(&out[..n*C]);
    let frame=encode(&mr,&pair)?;let(_,input)=decode_query(&frame)?;
    let before=ic_cdk::api::performance_counter(0);let(y,b)=evaluate_decoded(&mr,input)?;reads+=b;spans.push(("next_mlp_front".into(),ic_cdk::api::performance_counter(0)-before));
    mr.step=mr.step.checked_add(1).ok_or("bridge progress overflow")?;
    let state=y.encode(&mr,cfg!(feature="experimental-host-checksum"))?;
    let previous_hidden=bf16_bytes(&both[..n*C]);let conv=bf16_bytes(&out[n*C..]);
    #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
    #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
    Ok(MlpDeltaMeasurement{state:state.into(),previous_hidden:previous_hidden.into(),conv:conv.into(),instructions:ic_cdk::api::performance_counter(0)-start,stable_read_bytes:reads,heap_pages,stable_pages:ic_cdk::api::stable_size(),spans})
}

#[cfg(test)]
mod tests {
    #[test]
    fn query_feature_enables_required_stream_codec() {
        let request = imajev_runtime::Request {
            version: 1, model: "a".repeat(64), pack_hash: "b".repeat(64),
            input_hash: "c".repeat(64), step: 0,
            op: "mlp_stream_prepare".into(), encoding: "mlp-stream-exact-v1".into(),
            tensor: "model.language_model.layers.1.post_attention_layernorm.weight".into(),
            aux: vec!["model.language_model.layers.2.input_layernorm.weight".into()],
            dims: vec![1, 0, 256], scalars: vec![2., 1e-6],
        };
        let frame = imajev_runtime::encode(&request, &vec![0.; 2 * super::C]).unwrap();
        let (_, input) = imajev_runtime::decode_query(&frame).unwrap();
        assert!(matches!(input, imajev_runtime::DecodedQueryInput::MlpStream(_)));
    }
}
