use std::{env,fs};
fn main()->Result<(),Box<dyn std::error::Error>> {let a:Vec<_>=env::args().collect();match a[1].as_str(){
 "query"=>fs::write(&a[5],candid::encode_args((fs::read(&a[2])?,a[3].parse::<u32>()?,a[4].parse::<bool>()?))?)?,
 "decode"=>{let raw=fs::read_to_string(&a[2])?;let raw=raw.trim().strip_prefix("0x").unwrap_or(raw.trim());let bytes:Vec<_>=(0..raw.len()).step_by(2).map(|i|u8::from_str_radix(&raw[i..i+2],16)).collect::<Result<_,_>>()?;if a[3]=="measurement" {let m:imajev_activation_bench::Measurement=candid::decode_one(&bytes)?;println!("{}",serde_json::to_string(&m)?);}else{let m:imajev_activation_bench::Preparation=candid::decode_one(&bytes)?;println!("{}",serde_json::to_string(&m)?);}},
 _=>return Err("query/decode expected".into())}Ok(())}
