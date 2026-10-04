//! Diagnostic: compile the actual restore source without unrelated model kernels.
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::RefCell;
pub type Result<T> = std::result::Result<T,String>;
#[path="../../../crates/imajev-runtime/src/delta_restore.rs"]pub mod delta_restore;
thread_local!{static OWNER:RefCell<Principal>=const{RefCell::new(Principal::anonymous())};}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|*o.borrow_mut()=owner);}
#[derive(CandidType,Serialize,Deserialize)]pub struct Measurement{pub digest:Vec<u8>,pub instructions:u64,pub values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn restore(input:Vec<u8>,key_major:bool)->Measurement {
 OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));assert_eq!(input.len(),45*6176*4);
 let begin=ic_cdk::api::performance_counter(0);
 let log:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
 let state=if key_major{delta_restore::restore_key_major(45,32,&log).unwrap()}else{delta_restore::restore(45,32,&log).unwrap()};
 let end=ic_cdk::api::performance_counter(0);let mut hash=Sha256::new();for v in &state{hash.update(v.to_le_bytes());}
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Measurement{digest:hash.finalize().to_vec(),instructions:end-begin,values:state.len()as u64,heap_pages}
}
#[cfg(test)]mod tests{use super::*;
 #[test]fn included_restore_does_not_need_model_recurrence(){let x=vec![0.;2*(128+256+2)];assert_eq!(delta_restore::restore(2,2,&x).unwrap(),vec![0.;2*128*128]);assert_eq!(delta_restore::restore_key_major(2,2,&x).unwrap(),vec![0.;2*128*128]);}
}
