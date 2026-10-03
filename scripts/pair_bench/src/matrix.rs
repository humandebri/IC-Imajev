//! Fixed F32 weights and same-Wasm original/tail comparisons, diagnostic only.
use super::{FIXED,Preparation,Measurement};
use std::cell::RefCell;
use sha2::{Digest,Sha256};
struct Weight {rows:usize,cols:usize,values:Vec<f32>}
thread_local! {static WEIGHT:RefCell<Option<Weight>>=const{RefCell::new(None)};}
fn owner() {FIXED.with(|s|assert_eq!(s.borrow().as_ref().unwrap().owner,ic_cdk::api::msg_caller()));}
#[ic_cdk::update]
fn matrix_begin(rows:u32,cols:u32)->Preparation {
    owner();let (r,c)=(rows as usize,cols as usize);assert!(r>0&&c>0&&r<=9216&&c<=9216&&r.checked_mul(c).is_some_and(|v|v<=900000));
    WEIGHT.with(|s|*s.borrow_mut()=Some(Weight{rows:r,cols:c,values:Vec::with_capacity(r*c)}));
    Preparation{instructions:0,bytes:0,rows:r as u64}
}
#[ic_cdk::update]
fn matrix_chunk(start:u32,bytes:Vec<u8>)->Preparation {
    owner();assert!(!bytes.is_empty()&&bytes.len()<=1_000_000&&bytes.len()%4==0);
    let v:Vec<_>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();assert!(v.iter().all(|v|v.is_finite()));
    WEIGHT.with(|s|{let mut s=s.borrow_mut();let w=s.as_mut().unwrap();assert_eq!(start as usize,w.values.len());assert!(w.values.len()+v.len()<=w.rows*w.cols);
        let begin=ic_cdk::api::performance_counter(0);w.values.extend(v);
        Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:bytes.len()as u64,rows:w.rows as u64}
    })
}
#[ic_cdk::query]
fn project_matrix(input:Vec<u8>,n:u32,bf16:bool,candidate:bool)->Measurement {
    owner();WEIGHT.with(|s|{
        let s=s.borrow();let w=s.as_ref().unwrap();assert_eq!(w.values.len(),w.rows*w.cols);
        let n=n as usize;let rows=if n>109&&w.rows>4096 {4096}else{w.rows};let width=if bf16 {2}else{4};
        assert!(n>0&&n<=132&&n*w.cols<=900000&&n*rows<=900000&&input.len()<=1_800_000&&input.len()==n*w.cols*width);
        let begin=ic_cdk::api::performance_counter(0);
        let x:Vec<_>=if bf16 {input.chunks_exact(2).map(|b|f32::from_bits((u16::from_le_bytes(b.try_into().unwrap())as u32)<<16)).collect()}
        else {input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()};assert!(x.iter().all(|v|v.is_finite()));
        let ready=ic_cdk::api::performance_counter(0);
        let out=if candidate {imajev_runtime::matrix_tail(&x,&w.values[..rows*w.cols],n,rows,w.cols).unwrap()}
        else {imajev_runtime::matrix_baseline(&x,&w.values[..rows*w.cols],n,rows,w.cols).unwrap()};
        let end=ic_cdk::api::performance_counter(0);assert!(out.iter().all(|v|v.is_finite()));
        let mut digest=Sha256::new();for v in &out {digest.update(v.to_le_bytes());}
        #[cfg(target_arch="wasm32")] let heap_pages=core::arch::wasm32::memory_size(0)as u64;
        #[cfg(not(target_arch="wasm32"))] let heap_pages=0;
        Measurement{digest:digest.finalize().to_vec(),quantize_instructions:0,input_prepare_instructions:ready-begin,project_instructions:end-ready,total_instructions:end-begin,output_values:out.len()as u64,heap_pages}
    })
}
