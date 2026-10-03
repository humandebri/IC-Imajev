//! Diagnostic only. All prefix state is supplied by the client in ordinary queries.
pub mod codec;
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::RefCell;
thread_local!{static OWNER:RefCell<Principal>=const{RefCell::new(Principal::anonymous())};}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|*o.borrow_mut()=owner);}
#[derive(CandidType,Serialize,Deserialize)]pub struct Measurement{pub digest:Vec<u8>,pub instructions:u64,pub values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn restore(input:Vec<u8>,hybrid:bool)->Measurement {
 OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));assert!(input.len()<=1_990_000);
 let begin=ic_cdk::api::performance_counter(0);
 let state=if hybrid {codec::decode(&input).unwrap()}else{assert_eq!(input.len(),45*6176*4);let log:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();imajev_runtime::delta_log::restore(45,32,&log).unwrap()};
 let end=ic_cdk::api::performance_counter(0);let mut hash=Sha256::new();for v in &state {hash.update(v.to_le_bytes());}
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Measurement{digest:hash.finalize().to_vec(),instructions:end-begin,values:state.len()as u64,heap_pages}
}

#[derive(CandidType,Serialize,Deserialize)]pub struct Preparation{pub state:Vec<u8>,pub digest:Vec<u8>,pub instructions:u64,pub heap_pages:u64}
/// Ordinary query: reconstruct and encode once, then return all reusable state.
#[ic_cdk::query]fn prepare_prefix(input:Vec<u8>)->Preparation {
 OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));assert_eq!(input.len(),45*6176*4);
 let begin=ic_cdk::api::performance_counter(0);
 let log:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
 let(packet,state)=codec::prepare(&log,45).unwrap();let end=ic_cdk::api::performance_counter(0);
 let mut hash=Sha256::new();for v in &state{hash.update(v.to_le_bytes());}
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Preparation{state:packet,digest:hash.finalize().to_vec(),instructions:end-begin,heap_pages}
}
