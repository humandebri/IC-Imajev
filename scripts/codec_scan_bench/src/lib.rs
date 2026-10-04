//! Same-module old finite+codec passes versus checked streaming codec.
use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::Cell;
pub type Result<T>=std::result::Result<T,String>;pub const MAX_FLOATS:usize=900_000;
#[path="../../../crates/imajev-runtime/src/bf16_codec.rs"]pub mod bf16_codec;
#[path="../../../crates/imajev-runtime/src/block_codec.rs"]pub mod block_codec;
mod old_block;
thread_local!{static OWNER:Cell<Option<Principal>>=const{Cell::new(None)};}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(Some(owner)));}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub instructions:u64,pub output_bytes:u64}
pub fn reference(input:&[u8],op:u8)->Result<Vec<u8>>{if op==0{let x=old_block::decode(input)?;if !x.iter().all(|v|v.is_finite()){return Err("invalid activation".into());}let mut b=vec![];old_block::append(&mut b,&x)?;Ok(b)}else if op==1{let x=old_block::decode(input)?;if !x.iter().all(|v|v.is_finite()){return Err("invalid activation".into());}Ok(x.iter().flat_map(|v|v.to_le_bytes()).collect())}else{Err("op".into())}}
#[ic_cdk::query]fn measure(input:Vec<u8>,op:u8,method:u8)->Result<Measurement>{
 OWNER.with(|o|assert_eq!(o.get(),Some(ic_cdk::api::msg_caller())));if input.len()>2_000_000||op>1||method>1{return Err("input bounds".into());}
 let x=if op==0{Some(block_codec::decode(&input)?)}else{None};let start=ic_cdk::api::performance_counter(0);
 let mut encoded=vec![];let mut decoded=vec![];
 if let Some(x)=x{if method==0{if !x.iter().all(|v|v.is_finite()){return Err("invalid activation".into());}old_block::append(&mut encoded,&x)?;}else{block_codec::append(&mut encoded,&x)?;}}
 else{decoded=if method==0{let x=old_block::decode(&input)?;if !x.iter().all(|v|v.is_finite()){return Err("invalid activation".into());}x}else{block_codec::decode(&input)?};}
 let instructions=ic_cdk::api::performance_counter(0)-start;
 if op==1{encoded=decoded.iter().flat_map(|v|v.to_le_bytes()).collect();}
 Ok(Measurement{digest:Sha256::digest(&encoded).to_vec(),instructions,output_bytes:encoded.len()as u64})}
