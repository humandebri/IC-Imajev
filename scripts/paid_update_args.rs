//! Candid encode/decode bridge for deterministic local paid inference evidence.
use candid::{CandidType,Deserialize,Principal};
use serde::Serialize;
use sha2::Digest;
#[path="../canisters/inference/src/paid_types.rs"]mod paid_types;
use paid_types::*;
#[derive(CandidType,Deserialize,Serialize)]struct Forward {target:Principal,method:String,args:Vec<u8>,cycles:u128}
#[derive(CandidType,Deserialize,Serialize)]struct ForwardResult {response:Result<Vec<u8>,String>,refunded:u128,balance_before:u128,balance_after:u128,instructions:u64}
#[derive(CandidType,Deserialize)]struct LocalCanister {canister_id:Principal}
#[derive(CandidType,Deserialize)]struct LocalUpload {canister_id:Principal,chunk:Vec<u8>}
#[derive(CandidType,Deserialize)]struct ChunkHash {hash:Vec<u8>}
#[derive(CandidType,Deserialize)]enum LocalUpgradeMode {#[allow(non_camel_case_types)] upgrade}
#[derive(CandidType,Deserialize)]struct LocalChunkedUpgrade {mode:LocalUpgradeMode,target_canister:Principal,store_canister:Option<Principal>,chunk_hashes_list:Vec<ChunkHash>,wasm_module_hash:Vec<u8>,arg:Vec<u8>,sender_canister_version:Option<u64>}
// Explicit local comparison ABI for the pinned older reference module only.
#[derive(CandidType,Deserialize)]struct HistoricalRequest {model:String,version:u32,token_ids:Vec<u32>,options:Vec<String>}
fn historical_request(j:&serde_json::Value)->HistoricalRequest {
 HistoricalRequest{model:j["model"].as_str().unwrap().to_string(),version:1,token_ids:serde_json::from_value(j["token_ids"].clone()).unwrap(),options:serde_json::from_value(j["options"].clone()).unwrap()}
}
fn encode_infer(j: &serde_json::Value) -> Vec<u8> {
 candid::encode_args((serde_json::from_value::<InferRequest>(j["request"].clone()).unwrap(),j["request_id"].as_str().unwrap().to_string())).unwrap()
}
fn main(){let a:Vec<String>=std::env::args().collect();let action=&a[1];let kind=&a[2];let input=std::fs::read_to_string(&a[3]).unwrap();
 if action=="args"{let j:serde_json::Value=serde_json::from_str(&input).unwrap();let b=match kind.as_str(){
 "quote"=>candid::encode_one(serde_json::from_value::<InferRequest>(j).unwrap()).unwrap(),
 "local_clear_chunks"=>candid::encode_one(LocalCanister{canister_id:Principal::from_text(j["canister"].as_str().unwrap()).unwrap()}).unwrap(),
 "local_upload_chunk"=>candid::encode_one(LocalUpload{canister_id:Principal::from_text(j["canister"].as_str().unwrap()).unwrap(),chunk:std::fs::read(j["file"].as_str().unwrap()).unwrap()}).unwrap(),
 "local_chunked_upgrade"=>candid::encode_one(LocalChunkedUpgrade{mode:LocalUpgradeMode::upgrade,target_canister:Principal::from_text(j["canister"].as_str().unwrap()).unwrap(),store_canister:None,chunk_hashes_list:serde_json::from_value::<Vec<Vec<u8>>>(j["hashes"].clone()).unwrap().into_iter().map(|hash|ChunkHash{hash}).collect(),wasm_module_hash:serde_json::from_value(j["module_hash"].clone()).unwrap(),arg:candid::encode_args(()).unwrap(),sender_canister_version:None}).unwrap(),
 "prepare"=>candid::encode_one(j.as_str().unwrap().to_string()).unwrap(),
 "upload_chunk"=>{let b:Vec<u8>=serde_json::from_value(j["bytes"].clone()).unwrap();let h=sha2::Sha256::digest(&b).to_vec();candid::encode_args((j["offset"].as_u64().unwrap(),b,h)).unwrap()},
 "hash_pack"=>candid::encode_one(j.as_u64().unwrap()).unwrap(),
 "infer"=>encode_infer(&j),
 "historical_quote"=>candid::encode_one(historical_request(&j)).unwrap(),
 "historical_infer"=>candid::encode_args((historical_request(&j["request"]),j["request_id"].as_str().unwrap().to_string())).unwrap(),
 "configure_paid"=>candid::encode_one(serde_json::from_value::<Config>(j).unwrap()).unwrap(),
 "inference_status"|"retry_inference_refund"=>candid::encode_one(j.as_str().unwrap().to_string()).unwrap(),
 "inference_step"=>candid::encode_args((j["job_id"].as_u64().unwrap(),j["stage"].as_u64().unwrap())).unwrap(),
 "paid_fault"=>candid::encode_args((j["stage"].as_u64(),j["trap"].as_bool().unwrap(),j["refund_fail"].as_bool().unwrap())).unwrap(),
 "paid_upgrade_probe"=>candid::encode_args((j["active"].as_bool().unwrap(),j["refund_state"].as_u64().unwrap()as u32)).unwrap(),
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
 "configure_paid"|"prepare"=>serde_json::to_value(candid::decode_one::<Result<(),String>>(&bytes).unwrap()).unwrap(),
 "upload_chunk"=>serde_json::to_value(candid::decode_one::<Result<u64,String>>(&bytes).unwrap()).unwrap(),
 "hash_pack"=>serde_json::to_value(candid::decode_one::<Result<(u64,bool),String>>(&bytes).unwrap()).unwrap(),
 "forward"=>serde_json::to_value(candid::decode_one::<ForwardResult>(&bytes).unwrap()).unwrap(),
 "balance"=>serde_json::to_value(candid::decode_one::<u128>(&bytes).unwrap()).unwrap(),
 "paid_fault"|"paid_reference"|"paid_upgrade_probe"=>serde_json::Value::Null,_=>panic!("unknown decode kind")};println!("{}",value);
 }
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn inference_wire_contains_only_request_and_id() {
  let j=serde_json::json!({"request":{"token_ids":[1,2],"options":["yes","no"]},"request_id":"test-id"});
  let b=encode_infer(&j);
  let (request,id):(InferRequest,String)=candid::decode_args(&b).unwrap();
  assert_eq!(request.token_ids,vec![1,2]);
  assert_eq!(id,"test-id");
  assert_eq!(b,candid::encode_args((request,id)).unwrap());
 }
 #[test]
 fn explicit_historical_request_preserves_tokens_and_is_separate_from_current_abi() {
  let j=serde_json::json!({"model":"a".repeat(64),"token_ids":[1,2],"options":["yes","no"]});
  let bytes=candid::encode_one(historical_request(&j)).unwrap();
  let old:HistoricalRequest=candid::decode_one(&bytes).unwrap();
  assert_eq!(old.model,"a".repeat(64));assert_eq!(old.version,1);
  let current:InferRequest=candid::decode_one(&bytes).unwrap();
  assert_eq!(current.token_ids,vec![1,2]);assert_eq!(current.options,vec!["yes","no"]);
  assert_ne!(bytes,candid::encode_one(current).unwrap());
 }
}
