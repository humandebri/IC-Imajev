use std::{env,fs};use sha2::{Digest,Sha256};
fn main()->Result<(),Box<dyn std::error::Error>>{let a:Vec<_>=env::args().collect();match a[1].as_str(){
 "prepare"=>fs::write(&a[3],candid::encode_args((fs::read(&a[2])?,))?)?,
 "query"=>fs::write(&a[4],candid::encode_args((fs::read(&a[3])?,a[2].parse::<bool>()?))?)?,
 "decode"=>{let hex=fs::read_to_string(&a[2])?;let hex=hex.trim().strip_prefix("0x").unwrap_or(hex.trim());let b:Vec<u8>=(0..hex.len()).step_by(2).map(|i|u8::from_str_radix(&hex[i..i+2],16)).collect::<Result<_,_>>()?;let r:imajev_prefix_codec_bench::Measurement=candid::decode_one(&b)?;println!("{}",serde_json::to_string(&r)?);},
 "decode-preparation"=>{let hex=fs::read_to_string(&a[2])?;let hex=hex.trim().strip_prefix("0x").unwrap_or(hex.trim());let b:Vec<u8>=(0..hex.len()).step_by(2).map(|i|u8::from_str_radix(&hex[i..i+2],16)).collect::<Result<_,_>>()?;let r:imajev_prefix_codec_bench::Preparation=candid::decode_one(&b)?;fs::write(&a[3],&r.state)?;println!("{}",serde_json::json!({"digest":r.digest,"instructions":r.instructions,"heap_pages":r.heap_pages,"payload_bytes":r.state.len(),"payload_sha256":Sha256::digest(&r.state).iter().map(|v|format!("{v:02x}")).collect::<String>()}));},
 "native"=>{let b=fs::read(&a[2])?;let s=imajev_prefix_codec_bench::codec::decode(&b)?;let old=fs::read(&a[3])?;let bytes:Vec<u8>=s.iter().flat_map(|v|v.to_le_bytes()).collect();assert_eq!(bytes,old);println!("{}",serde_json::json!({"digest":Sha256::digest(bytes).to_vec(),"bitwise_equal":true}));},
 _=>return Err("query/decode/native expected".into())};Ok(())}
