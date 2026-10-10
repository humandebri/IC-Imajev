//! Candid encode/decode bridge for deterministic local paid inference evidence.
use candid::{CandidType,Deserialize,Principal};
use serde::Serialize;
#[path="../canisters/inference/src/paid_types.rs"]mod paid_types;
use paid_types::*;
#[derive(CandidType,Deserialize,Serialize)]struct Forward {target:Principal,method:String,args:Vec<u8>,cycles:u128}
#[derive(CandidType,Deserialize,Serialize)]struct ForwardResult {response:Result<Vec<u8>,String>,refunded:u128,balance_before:u128,balance_after:u128,instructions:u64}
fn encode_infer(j: &serde_json::Value) -> Vec<u8> {
 candid::encode_args((serde_json::from_value::<InferRequest>(j["request"].clone()).unwrap(),j["request_id"].as_str().unwrap().to_string())).unwrap()
}
fn main(){let a:Vec<String>=std::env::args().collect();let action=&a[1];let kind=&a[2];let input=std::fs::read_to_string(&a[3]).unwrap();
 if action=="args"{let j:serde_json::Value=serde_json::from_str(&input).unwrap();let b=match kind.as_str(){
 "quote"=>candid::encode_one(serde_json::from_value::<InferRequest>(j).unwrap()).unwrap(),
 "infer"=>encode_infer(&j),
 "configure_paid"=>candid::encode_one(serde_json::from_value::<Config>(j).unwrap()).unwrap(),
 "inference_status"|"retry_inference_refund"=>candid::encode_one(j.as_str().unwrap().to_string()).unwrap(),
 "inference_step"=>candid::encode_args((j["job_id"].as_u64().unwrap(),j["stage"].as_u64().unwrap())).unwrap(),
 "paid_fault"=>candid::encode_args((j["stage"].as_u64(),j["trap"].as_bool().unwrap(),j["refund_fail"].as_bool().unwrap())).unwrap(),
 "forward"=>candid::encode_one(Forward{target:Principal::from_text(j["target"].as_str().unwrap()).unwrap(),method:j["method"].as_str().unwrap().to_string(),args:std::fs::read(j["args"].as_str().unwrap()).unwrap(),cycles:j["cycles"].as_u64().unwrap()as u128}).unwrap(),
 "paid_reference"=>candid::encode_args((j.as_bool().unwrap(),)).unwrap(),
 "paid_debug"|"balance"|"paid_config"=>candid::encode_args(()).unwrap(),_=>panic!("unknown args kind")};std::fs::write(&a[4],b).unwrap();
 }else{let h=input.trim().trim_start_matches("0x");let bytes=(0..h.len()).step_by(2).map(|i|u8::from_str_radix(&h[i..i+2],16).unwrap()).collect::<Vec<_>>();
 let value=match kind.as_str(){
 "quote"=>serde_json::to_value(candid::decode_one::<Result<Quote,InferError>>(&bytes).unwrap()).unwrap(),
 "infer"=>serde_json::to_value(candid::decode_one::<Result<InferenceResult,InferError>>(&bytes).unwrap()).unwrap(),
 "inference_status"=>serde_json::to_value(candid::decode_one::<Option<Receipt>>(&bytes).unwrap()).unwrap(),
 "inference_step"=>serde_json::to_value(candid::decode_one::<Result<StepResult,String>>(&bytes).unwrap()).unwrap(),
 "retry_inference_refund"=>serde_json::to_value(candid::decode_one::<Result<Refund,String>>(&bytes).unwrap()).unwrap(),
 "paid_debug"=>serde_json::to_value(candid::decode_one::<Option<DebugResult>>(&bytes).unwrap()).unwrap(),
 "paid_config"=>serde_json::to_value(candid::decode_one::<Config>(&bytes).unwrap()).unwrap(),
 "configure_paid"=>serde_json::to_value(candid::decode_one::<Result<(),String>>(&bytes).unwrap()).unwrap(),
 "forward"=>serde_json::to_value(candid::decode_one::<ForwardResult>(&bytes).unwrap()).unwrap(),
 "balance"=>serde_json::to_value(candid::decode_one::<u128>(&bytes).unwrap()).unwrap(),
 "paid_fault"|"paid_reference"=>serde_json::Value::Null,_=>panic!("unknown decode kind")};println!("{}",value);
 }
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn inference_wire_contains_only_request_and_id() {
  let j=serde_json::json!({"request":{"model":"test","version":1,"token_ids":[1,2],"options":["yes","no"]},"request_id":"test-id"});
  let b=encode_infer(&j);
  let (request,id):(InferRequest,String)=candid::decode_args(&b).unwrap();
  assert_eq!(request.token_ids,vec![1,2]);
  assert_eq!(id,"test-id");
  assert_eq!(b,candid::encode_args((request,id)).unwrap());
 }
}
