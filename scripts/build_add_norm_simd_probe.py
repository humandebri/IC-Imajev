#!/usr/bin/env python3
"""Measure exact SIMD residual addition and normalization, preserving sum order."""
from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/add-norm-simd-v1/build';d.mkdir(parents=True,exist_ok=False)
 original=ROOT/'artifacts/update-conv4-cached-v2/build/runtime/lib.rs';s=original.read_text()
 rms=s[s.index('pub fn rms('):s.index('pub fn softmax(')]
 helper=r'''#[inline(always)]fn bf(v:f32)->f32{let b=v.to_bits();f32::from_bits(b.wrapping_add(0x7fff+((b>>16)&1))&0xffff0000)}
#[target_feature(enable="simd128")]
unsafe fn add_norm_simd(x:&[f32],w:&[f32],n:usize,c:usize,eps:f32)->Vec<f32>{
use core::arch::wasm32::*;
#[inline(always)]unsafe fn round(v:v128)->v128{let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))}
let count=n*c;let mut out=vec![0.;count*2];let mut i=0;
while i+4<=count{v128_store(out.as_mut_ptr().add(i).cast(),round(f32x4_add(v128_load(x.as_ptr().add(i).cast()),v128_load(x.as_ptr().add(count+i).cast()))));i+=4;}
while i<count{out[i]=bf(x[i]+x[count+i]);i+=1;}
for row in 0..n {
let start=row*c;let mut total=0f32;let mut j=0;
while j+4<=c{let v=v128_load(out.as_ptr().add(start+j).cast());let squares=f32x4_mul(v,v);
total+=f32x4_extract_lane::<0>(squares);total+=f32x4_extract_lane::<1>(squares);total+=f32x4_extract_lane::<2>(squares);total+=f32x4_extract_lane::<3>(squares);j+=4;}
while j<c{let v=out[start+j];total+=v*v;j+=1;}
let scale=(total/c as f32+eps).sqrt().recip();let sv=f32x4_splat(scale);j=0;
while j+4<=c{let v=v128_load(out.as_ptr().add(start+j).cast());let weight=v128_load(w.as_ptr().add(j).cast());v128_store(out.as_mut_ptr().add(count+start+j).cast(),round(f32x4_mul(f32x4_mul(v,sv),weight)));j+=4;}
while j<c{out[count+start+j]=bf(out[start+j]*scale*w[j]);j+=1;}
}out
}
'''
 (d/'add_norm_simd.rs').write_text(helper)
 header=r'''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use std::cell::Cell;
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{
OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=1&&input.len()>=24&&input.len()%4==0&&input.len()<=1_900_000);
let begin=ic_cdk::api::performance_counter(0);let v:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let n=v[0]as usize;let c=v[1]as usize;let eps=v[2];assert!(n>0&&c>0&&n*c<=220000&&v.len()==3+2*n*c+c&&eps.is_finite()&&eps>0.&&v.iter().all(|v|v.is_finite()));let x=&v[3..3+2*n*c];let w=&v[3+2*n*c..];let prepared=ic_cdk::api::performance_counter(0);
let y=if method==0{let count=n*c;let mut values:Vec<_>=(0..count).map(|i|bf(x[i]+x[count+i])).collect();let mut normalized=Vec::with_capacity(count);for row in values.chunks_exact(c){normalized.extend(rms(row,w,eps).into_iter().map(bf));}values.extend(normalized);values}else{unsafe{add_norm_simd(x,w,n,c,eps)}};
let end=ic_cdk::api::performance_counter(0);let digest=y.iter().flat_map(|v|v.to_le_bytes()).collect();Measurement{digest,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:y.len()as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
}
'''
 (d/'lib.rs').write_text(header+rms+helper)
 baseline=json.loads((ROOT/'artifacts/lora-finish-v1/build/report.json').read_text());command=baseline['command'][:];command[command.index('--crate-name')+1]='imajev_add_norm_simd_probe';command[command.index('--edition=2021')+1]=str(d/'lib.rs');command[command.index('-o')+1]=str(d/'diagnostic.wasm')
 with (d/'compiler.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=dict(os.environ,**baseline['explicit_env']),check=True,stdout=log,stderr=log)
 deps=[Path(command[i+1].split('=',1)[1]) for i,v in enumerate(command) if v=='--extern']
 r=dict(module=sha(d/'diagnostic.wasm'),command=command,explicit_env=baseline['explicit_env'],source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [d/'lib.rs',d/'add_norm_simd.rs',original,Path(__file__)]},dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in deps},scope='Original scalar add_norm expression and rms body copied verbatim; SIMD component products preserve scalar lane sum and F32 multiply order.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(r['module'])
if __name__=='__main__':main()
