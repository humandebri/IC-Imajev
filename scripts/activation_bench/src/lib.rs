//! Fixed math preparation only; ordinary queries retain no inference state.
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::RefCell;
use imajev_runtime::prepared_activation;
thread_local! {static OWNER:RefCell<Principal>=const{RefCell::new(Principal::anonymous())};}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|*o.borrow_mut()=owner);}
fn owner(){OWNER.with(|o|assert_eq!(*o.borrow(),ic_cdk::api::msg_caller()));}
#[derive(CandidType,Deserialize,Serialize)]pub struct Preparation {pub bytes:u64,pub instructions:u64}
#[ic_cdk::update]fn prepare()->Preparation {owner();let begin=ic_cdk::api::performance_counter(0);let bytes=prepared_activation::prepare();Preparation{bytes:bytes as u64,instructions:ic_cdk::api::performance_counter(0)-begin}}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement {pub digest:Vec<u8>,pub handler_instructions:u64,pub kernel_instructions:u64,pub values:u64,pub table_bytes:u64,pub heap_pages:u64}
fn apply<const K:usize>(x:&[f32],cached:bool)->Vec<f32> {if cached {x.iter().map(|v|prepared_activation::optimized::<K>(*v)).collect()}else{x.iter().map(|v|prepared_activation::original::<K>(*v)).collect()}}
#[ic_cdk::query]fn measure(input:Vec<u8>,kind:u32,cached:bool)->Measurement {owner();assert!(kind<4&&input.len()<=1_500_000&&input.len()%4==0);assert_eq!(prepared_activation::bytes(),1_048_576);let begin=ic_cdk::api::performance_counter(0);let x:Vec<_>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let kernel=ic_cdk::api::performance_counter(0);let y=match kind {0=>apply::<0>(&x,cached),1=>apply::<1>(&x,cached),2=>apply::<2>(&x,cached),3=>apply::<3>(&x,cached),_=>unreachable!()};let end=ic_cdk::api::performance_counter(0);let mut hash=Sha256::new();for v in &y {hash.update(v.to_le_bytes());}
 #[cfg(target_arch="wasm32")]let heap_pages=core::arch::wasm32::memory_size(0)as u64;
 #[cfg(not(target_arch="wasm32"))]let heap_pages=0;
 Measurement{digest:hash.finalize().to_vec(),handler_instructions:end-begin,kernel_instructions:end-kernel,values:y.len()as u64,table_bytes:prepared_activation::bytes()as u64,heap_pages}}
