//! Diagnostic byte decoder only. All carry bytes supplied by ordinary query.
pub type Result<T>=std::result::Result<T,String>;
mod baseline;
#[path="../../../crates/imajev-runtime/src/carry_planes.rs"]mod candidate;
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::RefCell;
thread_local!{static OWNER:RefCell<Principal>=const{RefCell::new(Principal::anonymous())};}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|*o.borrow_mut()=owner);}
#[derive(CandidType,Serialize,Deserialize)]pub struct Measurement{pub digest:Vec<u8>,pub instructions:u64,pub bytes:u64,pub heap_pages:u64}
#[ic_cdk::query]fn measure(input:Vec<u8>,n:u32,p:u32,optimized:bool)->Measurement {
 OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));assert!(input.len()<=1_990_000);assert!((1..=89).contains(&n));assert!((1..=132).contains(&p));let(n,p)=(n as usize,p as usize);
 let shapes=[(n*2560,2),(n*9216,1),(n*100,4),(24576,2),(p*2048,2),(p*4128,4)];let begin=ic_cdk::api::performance_counter(0);
 let groups=if optimized{candidate::decode(&input,&shapes)}else{baseline::decode(&input,&shapes)}.unwrap();let end=ic_cdk::api::performance_counter(0);
 let mut digest=Sha256::new();let mut bytes=0;for group in groups{bytes+=group.len()as u64;digest.update(group);}
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Measurement{digest:digest.finalize().to_vec(),instructions:end-begin,bytes,heap_pages}
}
