//! Isolated F32 LoRA A/B measurement. Only immutable owner-prepared weights persist.
pub mod packed;
#[cfg(target_arch="wasm32")]mod kernel;
use candid::{CandidType,Principal};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::cell::RefCell;
struct Fixed{owner:Principal,rows:usize,cols:usize,w:Vec<f32>,packed:Option<packed::Packed>}
thread_local!{static FIXED:RefCell<Option<Fixed>>=const{RefCell::new(None)};}
#[ic_cdk::init]fn init(owner:Principal,rows:u32,cols:u32){assert_ne!(owner,Principal::anonymous());assert!(rows>0&&rows<=9728&&rows%32==0&&cols>=64&&cols<=9216&&cols%64==0);FIXED.with(|f|*f.borrow_mut()=Some(Fixed{owner,rows:rows as usize,cols:cols as usize,w:vec![],packed:None}));}
#[derive(CandidType,Deserialize,Serialize)]pub struct Preparation{pub instructions:u64,pub bytes:u64,pub rows:u64}
#[ic_cdk::update]fn prepare_chunk(start:u32,bytes:Vec<u8>)->Preparation{FIXED.with(|s|{let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(f.packed.is_none()&&start as usize*f.cols==f.w.len()&&!bytes.is_empty()&&bytes.len()<=1_500_000&&bytes.len()%(f.cols*4)==0);let n=bytes.len()/(f.cols*4);assert!(start as usize+n<=f.rows);let begin=ic_cdk::api::performance_counter(0);for b in bytes.chunks_exact(4){let v=f32::from_le_bytes(b.try_into().unwrap());assert!(v.is_finite());f.w.push(v);}Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:bytes.len()as u64,rows:n as u64}})}
#[ic_cdk::update]fn seal()->Preparation{FIXED.with(|s|{let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(f.packed.is_none()&&f.w.len()==f.rows*f.cols);let begin=ic_cdk::api::performance_counter(0);f.packed=Some(packed::Packed::new(&f.w,f.rows,f.cols).unwrap());Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:f.packed.as_ref().unwrap().bytes()as u64,rows:f.rows as u64}})}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{FIXED.with(|s|{let s=s.borrow();let f=s.as_ref().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(f.packed.is_some()&&!input.is_empty()&&input.len()%(f.cols*4)==0&&method<=1);let n=input.len()/(f.cols*4);assert!(n>0&&n<=132&&n*f.rows<=900_000);let begin=ic_cdk::api::performance_counter(0);let x:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let prep=ic_cdk::api::performance_counter(0);let out=if method==0{imajev_runtime::matrix(&x,&f.w,n,f.rows,f.cols).unwrap()}else{f.packed.as_ref().unwrap().project(&x,n).unwrap()};let end=ic_cdk::api::performance_counter(0);let mut digest=Sha256::new();for v in &out{digest.update(v.to_le_bytes());}
#[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
#[cfg(not(target_arch="wasm32"))]let heap_pages=0;
Measurement{digest:digest.finalize().to_vec(),quantize_instructions:0,input_prepare_instructions:prep-begin,project_instructions:end-prep,total_instructions:end-begin,output_values:out.len()as u64,heap_pages}})}
