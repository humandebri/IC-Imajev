//! Offline verification only. Never used to advance live model inference.
use std::{env,fs};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=env::args().collect();if a.len()!=4 {return Err("tokens input-f32 output-f32 expected".into());}
 let n=a[1].parse()?;let bytes=fs::read(&a[2])?;if bytes.len()%4!=0 {return Err("F32 length".into());}
 let x:Vec<_>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let state=imajev_runtime::delta_log::restore(n,32,&x)?;
 fs::write(&a[3],state.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;Ok(())
}
