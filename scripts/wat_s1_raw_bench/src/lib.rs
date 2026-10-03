//! Dedicated diagnostic canister: fixed model weights are prepared once in updates.
//! No inference client state is retained and the production canister is untouched.
pub mod exact;
#[cfg(target_arch="wasm32")]
mod kernel;
use candid::{CandidType,Principal};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::cell::RefCell;
use imajev_runtime::int8_kernel;
struct Fixed {owner:Principal,rows:usize,cols:usize,w:Vec<i8>,scales:Vec<f32>,paired:Option<imajev_runtime::output_pairs::PreparedPairs>,coeff:Option<exact::Prepared>,sealed:bool}
thread_local! {static FIXED:RefCell<Option<Fixed>>=const{RefCell::new(None)};}
#[ic_cdk::init]
fn init(owner:Principal,rows:u32,cols:u32) {assert_ne!(owner,Principal::anonymous());assert!(rows>0 && rows<=9216 && rows%32==0 && cols>0 && cols<=9216 && cols%256==0);FIXED.with(|f|*f.borrow_mut()=Some(Fixed{owner,rows:rows as usize,cols:cols as usize,w:vec![],scales:vec![],paired:None,coeff:None,sealed:false}));}
#[derive(CandidType,Deserialize,Serialize)]
pub struct Preparation {pub instructions:u64,pub bytes:u64,pub rows:u64}
#[ic_cdk::update]
fn prepare_chunk(start:u32,bytes:Vec<u8>)->Preparation {
 FIXED.with(|s| {let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(!f.sealed);assert_eq!(start as usize,f.scales.len());assert!(!bytes.is_empty() && bytes.len()<=1_500_000 && bytes.len()%(f.cols+4)==0);
 let count=bytes.len()/(f.cols+4);assert!(f.scales.len()+count<=f.rows);let begin=ic_cdk::api::performance_counter(0);
 let weights=&bytes[..count*f.cols];for &v in weights {f.w.push(v as i8);}for b in bytes[count*f.cols..].chunks_exact(4){let v=f32::from_le_bytes(b.try_into().unwrap());assert!(v.is_finite()&&v>0.);f.scales.push(v);}
 Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:bytes.len() as u64,rows:count as u64}
 })
}
#[ic_cdk::update]
fn seal()->Preparation {FIXED.with(|s|{let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(!f.sealed && f.scales.len()==f.rows);let begin=ic_cdk::api::performance_counter(0);let mut bytes:Vec<u8>=f.w.iter().map(|v|*v as u8).collect();for v in &f.scales{bytes.extend(v.to_le_bytes());}f.paired=Some(imajev_runtime::output_pairs::PreparedPairs::from_le_bytes(&bytes,f.rows,f.cols).unwrap());f.coeff=Some(exact::Prepared::new(&f.w,f.rows,f.cols).unwrap());f.sealed=true;Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:(f.paired.as_ref().unwrap().bytes()+f.coeff.as_ref().unwrap().bytes())as u64,rows:f.rows as u64}})}
#[derive(CandidType,Deserialize,Serialize)]
pub struct Measurement {pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]
fn project(input:Vec<u8>,method:u8)->Measurement {FIXED.with(|s| {let s=s.borrow();let f=s.as_ref().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(f.sealed && !input.is_empty() && input.len()<=1_500_000 && input.len()%(f.cols*4)==0);let n=input.len()/(f.cols*4);let rows=if n>109 {f.rows.min(4096)} else {f.rows};assert!(n>0 && n<=132 && n*rows<=900_000);let begin=ic_cdk::api::performance_counter(0);
 let x:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let q=int8_kernel::quantize_rows(&x,n,f.cols).unwrap();let quant=ic_cdk::api::performance_counter(0);assert!(method<=1);let prepared=(method>0).then(||exact::operands(&q));let prep=ic_cdk::api::performance_counter(0);
 let out=if let Some(prepared)=prepared {f.coeff.as_ref().unwrap().project(&q,&prepared,&f.scales[..rows],rows).unwrap()}else{let view=f.paired.as_ref().unwrap().packed_rows(0,rows).unwrap();imajev_runtime::output_pairs::project(&q,&view,&f.scales[..rows]).unwrap()};let end=ic_cdk::api::performance_counter(0);
 let mut digest=Sha256::new();for v in &out {digest.update(v.to_le_bytes());}
 #[cfg(target_arch="wasm32")] let heap_pages=core::arch::wasm32::memory_size(0) as u64;
 #[cfg(not(target_arch="wasm32"))] let heap_pages=0;
 Measurement {digest:digest.finalize().to_vec(),quantize_instructions:quant-begin,input_prepare_instructions:prep-quant,project_instructions:end-prep,total_instructions:end-begin,output_values:out.len()as u64,heap_pages}
 })}
