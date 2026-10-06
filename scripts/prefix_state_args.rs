use std::{env,fs};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=env::args().collect();match a[1].as_str(){
 "args"=>fs::write(&a[4],candid::encode_args((fs::read(&a[2])?,fs::read(&a[3])?))?)?,
 "decode"=>{let text=fs::read_to_string(&a[2])?;let text=text.trim().strip_prefix("0x").unwrap_or(text.trim());let bytes=(0..text.len()).step_by(2).map(|i|u8::from_str_radix(&text[i..i+2],16)).collect::<Result<Vec<_>,_>>()?;
 let reply:Result<(u32,u64,u64),String>=candid::decode_one(&bytes)?;
 match reply{Ok((count,bytes,instructions))=>println!("{{\"count\":{count},\"bytes\":{bytes},\"instructions\":{instructions}}}"),Err(e)=>return Err(e.into())}},
 _=>return Err("args/decode expected".into())
 }Ok(())
}
