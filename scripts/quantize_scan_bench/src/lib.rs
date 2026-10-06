//! Same-module control and one-pass finite/maximum scan, ordinary query only.
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::Cell;
pub type Result<T>=std::result::Result<T,String>;
#[cfg(target_arch="wasm32")]mod old;
#[path="../../../crates/imajev-runtime/src/quantize_simd.rs"]mod candidate;
thread_local!{static OWNER:Cell<Option<Principal>>=const{Cell::new(None)};}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(Some(owner)));}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub instructions:u64,pub values:u64,pub scales:u64}
pub fn reference(x:&[f32],n:usize,c:usize)->Result<Vec<u8>>{let(q,s)=quantize(x,n,c,0)?;Ok(bytes(&q,&s))}
fn bytes(q:&[i16],s:&[f32])->Vec<u8>{let mut b=Vec::with_capacity(q.len()*2+s.len()*4);for v in q{b.extend(v.to_le_bytes());}for v in s{b.extend(v.to_le_bytes());}b}
fn quantize(x:&[f32],n:usize,c:usize,method:u8)->Result<(Vec<i16>,Vec<f32>)>{
 if n==0 || n>512 || c==0 || c>9216 || c%256!=0 || n.checked_mul(c)!=Some(x.len()) || method>1{return Err("shape".into());}
 if method==0 && !x.iter().all(|v|v.is_finite()){return Err("integer projection input".into());}
 let pad=n.div_ceil(8)*8;let mut q=vec![0i16;pad*c];let mut s=vec![1.;pad*c/256];
 for start in (0..x.len()).step_by(256){
  #[cfg(target_arch="wasm32")]unsafe{s[start/256]=if method==0{old::block(x.as_ptr().add(start),q.as_mut_ptr().add(start))}else{candidate::block(x.as_ptr().add(start),q.as_mut_ptr().add(start))?};}
  #[cfg(not(target_arch="wasm32"))]{let v=&x[start..start+256];let max=if method==0{v.iter().map(|v|v.abs()).fold(0f32,f32::max)}else{let bits=v.iter().map(|v|v.to_bits()&0x7fffffff).max().unwrap();if bits>=0x7f800000{return Err("integer projection input".into());}f32::from_bits(bits)};let scale=if max==0.{1.}else{(max/127.).max(f32::from_bits(1))};s[start/256]=scale;for i in 0..256{q[start+i]=(v[i]/scale).round_ties_even().clamp(-127.,127.)as i16;}}
 }Ok((q,s))}
#[ic_cdk::query]fn measure(input:Vec<u8>,cols:u32,method:u8)->Result<Measurement>{
 OWNER.with(|o|assert_eq!(o.get(),Some(ic_cdk::api::msg_caller())));if input.is_empty()||input.len()>1_900_000||input.len()%4!=0||cols==0{return Err("input bounds".into());}
 let x:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let c=cols as usize;if x.len()%c!=0{return Err("shape".into());}
 let begin=ic_cdk::api::performance_counter(0);let(q,s)=quantize(&x,x.len()/c,c,method)?;let instructions=ic_cdk::api::performance_counter(0)-begin;
 Ok(Measurement{digest:Sha256::digest(bytes(&q,&s)).to_vec(),instructions,values:q.len()as u64,scales:s.len()as u64})}
#[cfg(test)]mod tests{use super::*;#[test]fn finite_extremes_ties_and_subnormals_match(){for n in [1,7,8,45,87,89,132,512]{let x:Vec<f32>=(0..n*512).map(|i|[0.,-0.,f32::MAX,-f32::MAX,f32::from_bits(1),-f32::from_bits(1),0.5,-0.5,127.,-127.][i%10]).collect();assert_eq!(bytes(&quantize(&x,n,512,0).unwrap().0,&quantize(&x,n,512,0).unwrap().1),bytes(&quantize(&x,n,512,1).unwrap().0,&quantize(&x,n,512,1).unwrap().1));}}
#[test]fn all_nan_bits_and_late_failure_rejected(){for bits in [0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xff800001,0xffffffff]{for at in [0,255,256,511,1535]{let mut x=vec![0.;1536];x[at]=f32::from_bits(bits);for mode in [0,1]{assert!(quantize(&x,3,512,mode).is_err());}}}}}
