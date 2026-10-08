use std::{env,fs};
use sha2::{Digest,Sha256};
fn main()->Result<(),Box<dyn std::error::Error>> {let a:Vec<_>=env::args().collect();match a[1].as_str(){
 "matrix_begin"=>{fs::write(&a[4],candid::encode_args((a[2].parse::<u32>()?,a[3].parse::<u32>()?))?)?;},
 "matrix_query"=>{fs::write(&a[6],candid::encode_args((fs::read(&a[2])?,a[3].parse::<u32>()?,a[4].parse::<bool>()?,a[5].parse::<bool>()?))?)?;},
 #[cfg(feature="matrix-tail")]
 "matrix_native"|"matrix_ax"=>{
  let(n,rows,cols)=(a[2].parse::<usize>()?,a[3].parse::<usize>()?,a[4].parse::<usize>()?);
  let floats=|p:&str|->Result<Vec<f32>,Box<dyn std::error::Error>>{let b=fs::read(p)?;assert_eq!(b.len()%4,0);Ok(b.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect())};
  let w=floats(&a[5])?;let x=floats(&a[6])?;
  let old=imajev_runtime::matrix_reference(&x,&w,n,rows,cols)?;
  let baseline=imajev_runtime::matrix_baseline(&x,&w,n,rows,cols)?;
  let new=imajev_runtime::matrix_tail(&x,&w,n,rows,cols)?;
  assert!(old.iter().zip(&baseline).all(|(a,b)|a.to_bits()==b.to_bits()));assert!(old.iter().zip(&new).all(|(a,b)|a.to_bits()==b.to_bits()));
  if a[1]=="matrix_ax"{fs::write(&a[7],old.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;}
  let mut digest=Sha256::new();for v in &old {digest.update(v.to_le_bytes());}println!("{}",serde_json::json!({"digest":digest.finalize().to_vec(),"output_values":old.len(),"bitwise_equal":true}));
 },
 "chunk"=>{let bytes=fs::read(&a[3])?;fs::write(&a[4],candid::encode_args((a[2].parse::<u32>()?,bytes))?)?;},
 "query"=>{let bytes=fs::read(&a[3])?;fs::write(&a[4],candid::encode_args((bytes,a[2].parse::<bool>()?))?)?;},
 "decode"=>{let raw=fs::read_to_string(&a[2])?;let raw=raw.trim().strip_prefix("0x").unwrap_or(raw.trim());let b:Vec<u8>=(0..raw.len()).step_by(2).map(|i|u8::from_str_radix(&raw[i..i+2],16)).collect::<Result<_,_>>()?;if a[3]=="measurement"{let v:imajev_pair_bench::Measurement=candid::decode_one(&b)?;println!("{}",serde_json::to_string(&v)?);}else{let v:imajev_pair_bench::Preparation=candid::decode_one(&b)?;println!("{}",serde_json::to_string(&v)?);}},
 "native"=>{let n=a[2].parse::<usize>()?;let rows=a[3].parse::<usize>()?;let cols=a[4].parse::<usize>()?;let bytes=fs::read(&a[5])?;assert_eq!(bytes.len(),rows*(cols+4));let w:Vec<i8>=bytes[..rows*cols].iter().map(|&b|b as i8).collect();let sw:Vec<f32>=bytes[rows*cols..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let x=fs::read(&a[6])?;assert_eq!(x.len(),n*cols*4);let x:Vec<f32>=x.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols)?;let old=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows)?;let(p,f)=imajev_pair_bench::pair::prepare(&w,cols);let(a,b)=imajev_pair_bench::pair::activation(&q);let new=imajev_pair_bench::pair::project(&q,&a,&b,&p,&f,&sw,rows);assert!(old.iter().zip(&new).all(|(a,b)|a.to_bits()==b.to_bits()));let mut digest=Sha256::new();for v in &old {digest.update(v.to_le_bytes());}println!("{}",serde_json::json!({"digest":digest.finalize().to_vec(),"output_values":old.len(),"bitwise_equal":true}));},
 _=>return Err("chunk/query/decode/native expected".into())};Ok(())}
