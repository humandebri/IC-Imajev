//! Offline Candid encoding/decoding only. No keys, agents, or network calls.
use candid::{CandidType, Decode, Deserialize, Encode};
use serde::Serialize;
use serde_json::{json, Value};
use std::{error::Error, fs, io::{self, Read}};
#[derive(CandidType, Deserialize, Serialize)]
struct PackStatus { model:String, pack_hash:String, bytes:u64, received:u64, hashed:u64, ready:bool, chunks:Vec<u64> }
#[derive(CandidType, Deserialize, Serialize)]
struct WeightCacheInfo { bytes:u64, names:Vec<String>, preparation_instructions:u64, rope_bytes:Option<u64>, activation_bytes:Option<u64>, paired_weight_bytes:Option<u64> }
#[derive(CandidType, Deserialize, Serialize)]
struct ChoiceResult { value:Option<String>, probabilities:Vec<f32>, unknown_probability:f32, abstained:bool, raw_logits:Vec<f32>, instructions:u64, calibration_version:String }
#[derive(CandidType, Deserialize, Serialize)]
struct UpdateProgress { id:u64, stage:u64, done:bool, instructions:u64, stable_read_bytes:u64, operations:Vec<(String,u64)>, hidden_hashes:Vec<String>, state_hashes:Vec<String>, final_hidden:Vec<f32>, decision:Option<ChoiceResult>, heap_pages:u64 }
#[derive(CandidType, Deserialize, Serialize)]
struct Config { base_fee:u128, fee_per_token:u128, version:u64, enabled:bool, reserve_cycles:u128 }
fn number(v:&Value)->Result<u64,Box<dyn Error>> { if let Some(n)=v.as_u64(){Ok(n)}else if let Some(s)=v.as_str(){Ok(s.parse()?)}else{Err("unsigned number required".into())} }
fn hex(s:&str)->Result<Vec<u8>,Box<dyn Error>> { let s=s.trim().strip_prefix("0x").unwrap_or(s.trim()); if s.len()%2!=0{return Err("odd hex length".into())} (0..s.len()).step_by(2).map(|i|Ok(u8::from_str_radix(&s[i..i+2],16)?)).collect() }
fn blob(v:&Value)->Result<Vec<u8>,Box<dyn Error>> { if let Some(p)=v.get("file").and_then(Value::as_str){Ok(fs::read(p)?)}else if let Some(s)=v.get("hex").and_then(Value::as_str){hex(s)}else if let Some(a)=v.as_array(){a.iter().map(|x|Ok(u8::try_from(number(x)?)?)).collect()}else{Err("blob file/hex/array required".into())} }
fn floats(v:&Value)->Result<Vec<f32>,Box<dyn Error>> { let out=if let Some(p)=v.get("f32_file").and_then(Value::as_str){let b=fs::read(p)?; if b.len()%4!=0{return Err("F32 file length".into())} b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>()}else{v.as_array().ok_or("F32 array/file required")?.iter().map(|x|Ok(x.as_f64().ok_or("float required")? as f32)).collect::<Result<Vec<_>,Box<dyn Error>>>()?}; if !out.iter().all(|x|x.is_finite()){return Err("nonfinite F32".into())}Ok(out) }
fn encode(method:&str,a:&[Value])->Result<Vec<u8>,Box<dyn Error>> {
    Ok(match method {
        "status"|"pack_status"|"weight_cache_status"|"paid_config"|"seal" => {if !a.is_empty(){return Err("zero args expected".into())}Encode!()?},
        "prepare"|"warm_weights" => {if a.len()!=1{return Err("one text arg expected".into())}Encode!(&a[0].as_str().ok_or("text required")?.to_owned())?},
        "hash_pack" => {if a.len()!=1{return Err("one nat64 expected".into())}Encode!(&number(&a[0])?)?},
        "upload_chunk" => {if a.len()!=3{return Err("three upload args expected".into())}{let offset=number(&a[0])?;let data=blob(&a[1])?;let digest=blob(&a[2])?;Encode!(&offset,&data,&digest)?}},
        "prepare_fixed_prefix_state" => {if a.len()!=2{return Err("two blobs expected".into())}{let state=blob(&a[0])?;let packet=blob(&a[1])?;Encode!(&state,&packet)?}},
        "update_prefix" => {if a.len()!=3{return Err("three prefix args expected".into())}{let layer=u32::try_from(number(&a[0])?)?;let values=floats(&a[1])?;let packet=blob(&a[2])?;Encode!(&layer,&values,&packet)?}},
        "update_infer_start" => {if a.len()!=2{return Err("token ids and options expected".into())}let ids=a[0].as_array().ok_or("ids required")?.iter().map(|x|Ok(u32::try_from(number(x)?)?)).collect::<Result<Vec<_>,Box<dyn Error>>>()?;let options=a[1].as_array().ok_or("options required")?.iter().map(|x|Ok(x.as_str().ok_or("option text required")?.to_owned())).collect::<Result<Vec<_>,Box<dyn Error>>>()?;Encode!(&ids,&options)?},
        "update_infer_continue" => {if a.len()!=2{return Err("id and stage expected".into())}{let id=number(&a[0])?;let stage=number(&a[1])?;Encode!(&id,&stage)?}},
        _=>return Err(format!("unsupported encode method {method}").into())
    })
}
fn decode(method:&str,b:&[u8])->Result<Value,Box<dyn Error>> {
    Ok(match method {
        "status"=>{let (n,r)=Decode!(b,u64,bool)?;json!([n,r])},
        "pack_status"=>serde_json::to_value(Decode!(b,PackStatus)?)?,
        "weight_cache_status"=>serde_json::to_value(Decode!(b,WeightCacheInfo)?)?,
        "paid_config"=>serde_json::to_value(Decode!(b,Config)?)?,
        "prepare"|"seal"|"update_prefix"=>serde_json::to_value(Decode!(b,Result<(),String>)?)?,
        "upload_chunk"=>serde_json::to_value(Decode!(b,Result<u64,String>)?)?,
        "hash_pack"=>serde_json::to_value(Decode!(b,Result<(u64,bool),String>)?)?,
        "warm_weights"=>serde_json::to_value(Decode!(b,Result<WeightCacheInfo,String>)?)?,
        "prepare_fixed_prefix_state"=>serde_json::to_value(Decode!(b,Result<(u32,u64,u64),String>)?)?,
        "update_infer_start"|"update_infer_continue"=>serde_json::to_value(Decode!(b,Result<UpdateProgress,String>)?)?,
        _=>return Err(format!("unsupported decode method {method}").into())
    })
}
fn main()->Result<(),Box<dyn Error>> {
    let mut s=String::new();io::stdin().read_to_string(&mut s)?;let v:Value=serde_json::from_str(&s)?;let method=v["method"].as_str().ok_or("method required")?;
    let out=match v["op"].as_str().ok_or("op required")? {
        "encode"=>{let b=encode(method,v["args"].as_array().ok_or("args required")?)?;let p=v["output"].as_str().ok_or("output required")?;if let Ok(existing)=fs::read(p){if existing!=b{return Err("existing args differ".into())}}else{fs::write(p,&b)?}json!({"bytes":b.len(),"output":p})},
        "decode"=>json!({"result":decode(method,&hex(v["hex"].as_str().ok_or("hex required")?)?)?}),
        _=>return Err("encode/decode required".into())
    };println!("{}",serde_json::to_string(&out)?);Ok(())
}
