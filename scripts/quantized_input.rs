//! Offline capacity diagnostic only. Never feeds host results into inference.
use std::{env,fs};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=env::args().collect();let n:usize=a[1].parse()?;let cols:usize=a[2].parse()?;
 let raw=fs::read(&a[3])?;assert_eq!(raw.len(),n*cols*4);
 let x:Vec<f32>=raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
 let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols)?;
 fs::write(&a[4],q.values()[..n*cols].iter().map(|&v|v as i8 as u8).collect::<Vec<_>>())?;
 Ok(())
}
