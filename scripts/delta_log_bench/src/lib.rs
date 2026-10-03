//! Isolated diagnostic only; no question state survives ordinary queries.
pub mod replay;
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::RefCell;
thread_local! {static OWNER:RefCell<Principal>=const{RefCell::new(Principal::anonymous())};}
#[ic_cdk::init] fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|x|*x.borrow_mut()=owner);}
#[derive(CandidType,Deserialize,Serialize)] pub struct Measurement {pub digest:Vec<u8>,pub handler_instructions:u64,pub kernel_instructions:u64,pub values:u64,pub heap_pages:u64}
#[ic_cdk::query] fn restore(input:Vec<u8>,tokens:u32,heads:u32,logged:bool)->Measurement {
 OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));assert!(input.len()<=1_900_000&&input.len()%4==0);
 let begin=ic_cdk::api::performance_counter(0);let x:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let kernel=ic_cdk::api::performance_counter(0);
 let result=if logged {replay::restore(tokens as usize,heads as usize,&x)}else{replay::reference(tokens as usize,heads as usize,&x)}.unwrap();let end=ic_cdk::api::performance_counter(0);
 let mut hash=Sha256::new();for v in &result {hash.update(v.to_le_bytes());}
 #[cfg(target_arch="wasm32")] let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))] let heap_pages=0;
 Measurement{digest:hash.finalize().to_vec(),handler_instructions:end-begin,kernel_instructions:end-kernel,values:result.len()as u64,heap_pages}
}
