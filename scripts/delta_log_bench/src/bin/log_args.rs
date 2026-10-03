use std::{env,fs,io::{Read,Seek,SeekFrom}};
use sha2::{Digest,Sha256};
use imajev_delta_log_bench::replay;
fn floats(p:&str)->Result<Vec<f32>,Box<dyn std::error::Error>> {let b=fs::read(p)?;assert_eq!(b.len()%4,0);Ok(b.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect())}
fn write_floats(p:&str,x:&[f32])->std::io::Result<()> {fs::write(p,x.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())}
fn main()->Result<(),Box<dyn std::error::Error>> {let a:Vec<_>=env::args().collect();match a[1].as_str(){
 "query"=>fs::write(&a[6],candid::encode_args((fs::read(&a[2])?,a[3].parse::<u32>()?,a[4].parse::<u32>()?,a[5].parse::<bool>()?))?)?,
 "decode"=>{let raw=fs::read_to_string(&a[2])?;let raw=raw.trim().strip_prefix("0x").unwrap_or(raw.trim());let b:Vec<_>=(0..raw.len()).step_by(2).map(|i|u8::from_str_radix(&raw[i..i+2],16)).collect::<Result<_,_>>()?;let m:imajev_delta_log_bench::Measurement=candid::decode_one(&b)?;println!("{}",serde_json::to_string(&m)?);},
 "native"=>{let n=a[3].parse()?;let h=a[4].parse()?;let x=floats(&a[2])?;let y=if a[5].parse::<bool>()? {replay::restore(n,h,&x)?}else{replay::reference(n,h,&x)?};let mut sha=Sha256::new();for v in &y {sha.update(v.to_le_bytes());}println!("{}",serde_json::json!({"digest":sha.finalize().to_vec(),"values":y.len()}));},
 "extract"=>{
 let (r,x)=imajev_runtime::decode(&fs::read(&a[2])?)?;assert_eq!(r.op,"delta_stage_bf16");let(n,h,first,keep)=(r.dims[0],r.dims[1],r.dims[2],r.dims[3]);assert_eq!(keep,1);
 let m:imajev_runtime::Manifest=serde_json::from_slice(&fs::read(&a[3])?)?;let tensor=m.tensors.iter().find(|t|t.name==r.tensor).unwrap();let mut file=fs::File::open(&a[4])?;
 let mut read=|off,len|{file.seek(SeekFrom::Start(off)).map_err(|e|e.to_string())?;let mut bytes=vec![0;len];file.read_exact(&mut bytes).map_err(|e|e.to_string())?;Ok(bytes)};
 let channels=h*256;let window=(n+3)*channels;let count=n*h*128;let mut cr=r.clone();cr.op="conv_state".into();cr.aux.clear();cr.scalars.clear();let mut weights=vec![];
 for(start,rows)in [(first/2*128,h/2*128),(2048+first/2*128,h/2*128),(4096+first*128,h*128)] {cr.dims=vec![n,rows,4,start];weights.extend(imajev_runtime::load_weight(tensor,&cr,&mut read)?.0);}
 cr.dims=vec![n,channels,4,0];let conv=imajev_runtime::execute(&cr,&x[..window],&weights)?;
 let mut nr=r.clone();nr.op="rms_scaled".into();nr.dims=vec![n,128];nr.aux.clear();nr.scalars=vec![1e-6,128f32.sqrt().recip()];
 let mut k=vec![0.;n*h/2*128];for head in 0..h/2 {let raw:Vec<_>=(0..n).flat_map(|t|conv[t*channels+h/2*128+head*128..t*channels+h/2*128+(head+1)*128].iter().copied()).collect();let v=imajev_runtime::execute(&nr,&raw,&[])?;for t in 0..n {k[(t*h/2+head)*128..(t*h/2+head+1)*128].copy_from_slice(&v[t*128..(t+1)*128]);}}
 let mut captured=k;for t in 0..n {captured.extend_from_slice(&conv[t*channels+h*128..(t+1)*channels]);}captured.extend_from_slice(&x[window+count..window+count+2*n*h]);
 assert!(x[window+count+2*n*h..].iter().all(|v|*v==0.));let(log,s)=replay::capture(n,h,&captured)?;let restored=replay::restore(n,h,&log)?;let reference=replay::reference(n,h,&captured)?;
 let(_,old)=imajev_runtime::decode(&fs::read(&a[5])?)?;assert_eq!(old.len(),count+h*16384);let expected=&old[count..];
 assert!(s.iter().zip(&restored).zip(&reference).zip(expected).all(|(((a,b),c),d)|a.to_bits()==b.to_bits()&&a.to_bits()==c.to_bits()&&a.to_bits()==d.to_bits()),"saved Wasm state mismatch");
 fs::create_dir_all(&a[6])?;write_floats(&format!("{}/capture.bin",a[6]),&captured)?;write_floats(&format!("{}/log.bin",a[6]),&log)?;write_floats(&format!("{}/state.bin",a[6]),&s)?;
 println!("{}",serde_json::json!({"tokens":n,"heads":h,"first":first,"bitwise_equal_to_saved_wasm":true,"state_values":s.len(),"log_values":log.len()}));
 },_=>return Err("extract/query/decode/native expected".into())}Ok(())}
