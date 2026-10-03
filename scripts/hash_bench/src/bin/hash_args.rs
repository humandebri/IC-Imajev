use std::{env,fs};
use sha2::{Digest,Sha256};
fn hex(b:&[u8])->String {b.iter().map(|v|format!("{v:02x}")).collect()}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<_>=env::args().collect();
    if a[1]=="args" {
        let data=fs::read(&a[2])?;
        if data.len()>2_000_000 {return Err("hash input limit".into());}
        fs::write(&a[3],candid::encode_one(&data)?)?;
        println!("{}",serde_json::json!({"input_bytes":data.len(),"sha256":hex(&Sha256::digest(&data)),"blake3":hex(blake3::hash(&data).as_bytes())}));
    } else if a[1]=="decode" {
        let raw=fs::read_to_string(&a[2])?;
        let raw=raw.trim().strip_prefix("0x").unwrap_or(raw.trim());
        let bytes:Vec<_>=(0..raw.len()).step_by(2).map(|i|u8::from_str_radix(&raw[i..i+2],16)).collect::<Result<_,_>>()?;
        let m:imajev_hash_bench::Measurement=candid::decode_one(&bytes)?;
        println!("{}",serde_json::json!({"digest":hex(&m.digest),"instructions":m.instructions,"input_bytes":m.input_bytes,"heap_pages":m.heap_pages,"reply_bytes":bytes.len()}));
    } else {return Err("args or decode expected".into());}
    Ok(())
}
