//! Isolated hash-only measurement; does not run inference or store model state.
use candid::{CandidType,Principal};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::cell::Cell;
thread_local! {static OWNER:Cell<Option<Principal>>=const {Cell::new(None)};}
#[ic_cdk::init]
fn init(owner:Principal) {assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(Some(owner)));}
fn check(input:&[u8]) {
    OWNER.with(|o|assert_eq!(o.get(),Some(ic_cdk::api::msg_caller()),"owner only"));
    assert!(input.len()<=2_000_000,"hash input limit");
}
#[derive(CandidType,Deserialize,Serialize)]
pub struct Measurement {
    pub digest:Vec<u8>,
    pub instructions:u64,
    pub input_bytes:u64,
    pub heap_pages:u64,
}
fn finish(digest:Vec<u8>,instructions:u64,len:usize)->Measurement {
    #[cfg(target_arch="wasm32")]
    let heap_pages=core::arch::wasm32::memory_size(0) as u64;
    #[cfg(not(target_arch="wasm32"))]
    let heap_pages=0;
    Measurement {digest,instructions,input_bytes:len as u64,heap_pages}
}
#[ic_cdk::query]
fn hash_sha256(input:Vec<u8>)->Measurement {
    check(&input);
    let start=ic_cdk::api::performance_counter(0);
    let digest=Sha256::digest(&input);
    let instructions=ic_cdk::api::performance_counter(0)-start;
    finish(digest.to_vec(),instructions,input.len())
}
#[ic_cdk::query]
fn hash_blake3(input:Vec<u8>)->Measurement {
    check(&input);
    let start=ic_cdk::api::performance_counter(0);
    let digest=blake3::hash(&input);
    let instructions=ic_cdk::api::performance_counter(0)-start;
    finish(digest.as_bytes().to_vec(),instructions,input.len())
}
