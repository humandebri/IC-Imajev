//! Diagnostic: compile the actual restore source without unrelated model kernels.
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::RefCell;
pub type Result<T> = std::result::Result<T,String>;
#[path="../../../crates/imajev-runtime/src/delta_restore.rs"]pub mod delta_restore;
#[path="../../../crates/imajev-runtime/src/carry_planes.rs"]pub mod carry_planes;
thread_local!{static OWNER:RefCell<Principal>=const{RefCell::new(Principal::anonymous())};}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|*o.borrow_mut()=owner);}
#[derive(CandidType,Serialize,Deserialize)]pub struct Measurement{pub digest:Vec<u8>,pub instructions:u64,pub values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn interleave_residual(input:Vec<u8>,legacy:bool)->Measurement {
 OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));assert!(input.len()>=4 && input.len()<=1_990_000);
 let count=u32::from_le_bytes(input[..4].try_into().unwrap())as usize;assert!(count>0 && count<=89*2560 && input.len()==4+2*count);
 let begin=ic_cdk::api::performance_counter(0);let p=&input[4..];let mut out=Vec::with_capacity(2*count);
 if legacy {for i in 0..count {out.push(p[i]);out.push(p[count+i]);}}
 else {out.resize(2*count,0);carry_planes::interleave(p,&mut out).unwrap();}
 let end=ic_cdk::api::performance_counter(0);
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Measurement{digest:Sha256::digest(&out).to_vec(),instructions:end-begin,values:out.len()as u64,heap_pages}
}
#[ic_cdk::query]fn decode_residual(input:Vec<u8>,_alternate:bool)->Measurement {
 OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));assert!(input.len()>=4 && input.len()<=1_990_000);
 let begin=ic_cdk::api::performance_counter(0);let count=u32::from_le_bytes(input[..4].try_into().unwrap())as usize;assert!(count>0 && count<=89*2560 && count%2560==0);
 let groups=carry_planes::decode(&input[4..],&[(count,2)]).unwrap();let end=ic_cdk::api::performance_counter(0);
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Measurement{digest:Sha256::digest(&groups[0]).to_vec(),instructions:end-begin,values:groups[0].len()as u64,heap_pages}
}
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
