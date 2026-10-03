//! Synthetic kernel diagnostic. Ordinary queries retain no inference state.
use candid::{CandidType,Principal};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::cell::RefCell;
// Compile the actual production Delta SIMD source in isolation, excluding
// unrelated INT8 projection kernels. Source hashes are archived by the checker.
#[cfg(target_arch="wasm32")]
#[path="../../../crates/imajev-runtime/src/delta_simd.rs"]
mod delta_simd;
fn compute(q:&[f32],k:&[f32],v:&[f32],g:&[f32],b:&[f32],state:&mut[f32],dk:usize,dv:usize,discard:bool)->Vec<f32>{
 #[cfg(target_arch="wasm32")]
 {unsafe {if discard {delta_simd::run::<false, false>(q,k,v,g,b,state,dk,dv)}else{delta_simd::run::<true, false>(q,k,v,g,b,state,dk,dv)}}}
 #[cfg(not(target_arch="wasm32"))]
 {if discard {imajev_runtime::delta_without_final_state(q,k,v,g,b,state,dk,dv)}else{imajev_runtime::delta(q,k,v,g,b,state,dk,dv)}.unwrap()}
}
thread_local! {static OWNER:RefCell<Principal>=const{RefCell::new(Principal::anonymous())};}
#[ic_cdk::init] fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|x|*x.borrow_mut()=owner);}
#[derive(CandidType,Deserialize,Serialize)]
pub struct Measurement {pub digest:Vec<u8>,pub kernel_instructions:u64,pub handler_instructions:u64,pub values:u64,pub scratch_unchanged:bool}
pub fn inputs(n:usize,dk:usize,dv:usize)->(Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>){
 let sequence=|len:usize,seed:u32,scale:f32| {let mut s=seed;(0..len).map(|_|{s=s.wrapping_mul(1664525).wrapping_add(1013904223);((s>>16)as i32-32768)as f32*(scale/32768.)}).collect::<Vec<_>>()};
 (sequence(n*dk,17,0.08),sequence(n*dk,53,0.08),sequence(n*dv,97,0.2),sequence(n,31,0.15).into_iter().map(|v|0.85+v).collect(),sequence(n,73,0.45).into_iter().map(|v|0.5+v).collect(),sequence(dk*dv,41,0.1))
}
#[ic_cdk::query] fn measure(n:u32,dk:u32,dv:u32,discard:bool)->Measurement {
 OWNER.with(|x|assert_eq!(*x.borrow(),ic_cdk::api::msg_caller()));
 assert!((1..=132).contains(&n)&&dk>0&&dk<=128&&dv>0&&dv<=128&&dv%4==0);
 let begin=ic_cdk::api::performance_counter(0);
 let(q,k,v,g,b,mut state)=inputs(n as usize,dk as usize,dv as usize);let initial=state.clone();
 let start=ic_cdk::api::performance_counter(0);
 let out=compute(&q,&k,&v,&g,&b,&mut state,dk as usize,dv as usize,discard);
 let end=ic_cdk::api::performance_counter(0);
 let mut hash=Sha256::new();for v in &out {hash.update(v.to_le_bytes());}
 Measurement{digest:hash.finalize().to_vec(),kernel_instructions:end-start,handler_instructions:end-begin,values:out.len()as u64,scratch_unchanged:initial.iter().zip(&state).all(|(a,b)|a.to_bits()==b.to_bits())}
}
/// Input layout conversion is preparation for this kernel-only diagnostic;
/// it is outside the kernel counter and still inside the handler counter.
#[ic_cdk::query] fn measure_layout(n:u32,dk:u32,dv:u32,key_major:bool)->Measurement {
 OWNER.with(|x|assert_eq!(*x.borrow(),ic_cdk::api::msg_caller()));
 assert!((1..=132).contains(&n)&&dk>0&&dk<=128&&dv>0&&dv<=128&&dv%4==0);
 let begin=ic_cdk::api::performance_counter(0);
 let(q,k,v,g,b,mut state)=inputs(n as usize,dk as usize,dv as usize);
 if key_major {let old=state.clone();for d in 0..dv as usize {for i in 0..dk as usize {state[i*dv as usize+d]=old[d*dk as usize+i];}}}
 let initial=state.clone();let start=ic_cdk::api::performance_counter(0);
 #[cfg(target_arch="wasm32")]
 let out=unsafe {if key_major {delta_simd::run::<false,true>(&q,&k,&v,&g,&b,&mut state,dk as usize,dv as usize)}else{delta_simd::run::<false,false>(&q,&k,&v,&g,&b,&mut state,dk as usize,dv as usize)}};
 #[cfg(not(target_arch="wasm32"))]
 let out=if key_major {imajev_runtime::delta_from_key_major(&q,&k,&v,&g,&b,&mut state,dk as usize,dv as usize)}else{imajev_runtime::delta_without_final_state(&q,&k,&v,&g,&b,&mut state,dk as usize,dv as usize)}.unwrap();
 let end=ic_cdk::api::performance_counter(0);let mut hash=Sha256::new();for v in &out {hash.update(v.to_le_bytes());}
 Measurement{digest:hash.finalize().to_vec(),kernel_instructions:end-start,handler_instructions:end-begin,values:out.len()as u64,scratch_unchanged:initial.iter().zip(&state).all(|(a,b)|a.to_bits()==b.to_bits())}
}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn output_contract_matches_existing_delta(){for n in [1,7,8,32,45,80,87,89,132]{for (dk,dv) in [(3,4),(128,12),(128,16),(128,128)]{
 let(q,k,v,g,b,mut s)=inputs(n,dk,dv);let mut t=s.clone();let expected=imajev_runtime::delta(&q,&k,&v,&g,&b,&mut s,dk,dv).unwrap();let actual=imajev_runtime::delta_without_final_state(&q,&k,&v,&g,&b,&mut t,dk,dv).unwrap();assert!(expected.iter().zip(actual).all(|(a,b)|a.to_bits()==b.to_bits()));
 }}}
 #[test] fn malformed_inputs_are_rejected(){assert!(imajev_runtime::delta_without_final_state(&[0.;4],&[0.;4],&[0.;4],&[1.1],&[0.5],&mut [0.;16],4,4).is_err());}
}
