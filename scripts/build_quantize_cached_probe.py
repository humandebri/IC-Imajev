#!/usr/bin/env python3
"""Compare original quantize SIMD with cached input registers and bounded normal-scale clamp elision."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/quantize-cached-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False);d=D/'build';d.mkdir()
 original=ROOT/'crates/imajev-runtime/src/quantize_simd.rs';baseline=original.read_text();(d/'baseline.rs').write_text(baseline)
 s='''#[target_feature(enable="simd128")]
pub unsafe fn block(x:*const f32,q:*mut i16,elide:bool)->crate::Result<f32>{
use core::arch::wasm32::*;let mask=i32x4_splat(0x7fffffff);let mut peak=i32x4_splat(0);
'''
 for k in range(64):s+=f'let v{k}=v128_load(x.add({k*4}).cast());peak=u32x4_max(peak,v128_and(v{k},mask));\n'
 s+='''peak=u32x4_max(peak,i32x4_shuffle::<2,3,0,1>(peak,peak));peak=u32x4_max(peak,i32x4_shuffle::<1,0,3,2>(peak,peak));
let bits=u32x4_extract_lane::<0>(peak);if bits>=0x7f800000{return Err("integer projection input".into());}
let max=f32::from_bits(bits);let scale=if max==0.{1.}else{(max/127.).max(f32::from_bits(1))};
let denominator=f32x4_splat(scale);let low=f32x4_splat(-127.);let high=f32x4_splat(127.);
let fast=elide&&scale>=f32::MIN_POSITIVE;
if fast {
'''
 for k in range(0,64,2):
  s+=f'let a=f32x4_nearest(f32x4_div(v{k},denominator));let b=f32x4_nearest(f32x4_div(v{k+1},denominator));v128_store(q.add({k*4}).cast(),i16x8_narrow_i32x4(i32x4_trunc_sat_f32x4(a),i32x4_trunc_sat_f32x4(b)));\n'
 s+='}else{\n'
 for k in range(0,64,2):
  s+=f'let a=f32x4_min(high,f32x4_max(low,f32x4_nearest(f32x4_div(v{k},denominator))));let b=f32x4_min(high,f32x4_max(low,f32x4_nearest(f32x4_div(v{k+1},denominator))));v128_store(q.add({k*4}).cast(),i16x8_narrow_i32x4(i32x4_trunc_sat_f32x4(a),i32x4_trunc_sat_f32x4(b)));\n'
 s+='}Ok(scale)}\n';(d/'candidate.rs').write_text(s)
 header='''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use std::cell::Cell;
type Result<T>=std::result::Result<T,String>;
mod baseline;mod candidate;
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{
OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=2&&!input.is_empty()&&input.len()%1024==0&&input.len()<=1900000);
let begin=ic_cdk::api::performance_counter(0);let x:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let prepared=ic_cdk::api::performance_counter(0);
let mut q=vec![0i16;x.len()];let mut scales=Vec::with_capacity(x.len()/256);let mut error=None;
for b in 0..x.len()/256{let value=unsafe{if method==0{baseline::block(x.as_ptr().add(b*256),q.as_mut_ptr().add(b*256))}else{candidate::block(x.as_ptr().add(b*256),q.as_mut_ptr().add(b*256),method==2)}};match value{Ok(s)=>scales.push(s),Err(e)=>{error=Some(e);break;}}}
let end=ic_cdk::api::performance_counter(0);let(digest,count)=if let Some(e)=error{let mut bytes=b"ERR:".to_vec();bytes.extend(e.as_bytes());(bytes,0)}else{let mut bytes:Vec<u8>=scales.iter().flat_map(|v|v.to_le_bytes()).collect();bytes.extend(q.iter().flat_map(|v|v.to_le_bytes()));(bytes,q.len()as u64)};
Measurement{digest,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:count,heap_pages:core::arch::wasm32::memory_size(0)as u64}}
'''
 (d/'lib.rs').write_text(header)
 eps=Fraction(1,2**24);bound=127*(1+eps)/(1-eps);assert bound<Fraction(255,2)
 (D/'normal-bound.json').write_text(json.dumps(dict(binary32_relative_rounding_bound='2^-24',upper_quotient_fraction=[bound.numerator,bound.denominator],rounds_at_most=127,guard='rounded scale >= minimum normal binary32',subnormal_scale='original division/nearest/clamp retained',nonfinite='absolute-bit peak rejects Inf/NaN before dividing',zero='scale1 and nearest0 preserved'),indent=2)+'\n')
 old=json.loads((ROOT/'artifacts/lora-finish-v1/build/report.json').read_text());cmd=old['command'][:];cmd[cmd.index('--crate-name')+1]='imajev_quantize_cached_probe';cmd[cmd.index('--edition=2021')+1]=str(d/'lib.rs');cmd[cmd.index('-o')+1]=str(d/'diagnostic.wasm')
 for p,h in old['dependency_hashes'].items():assert sha(ROOT/p)==h
 with(d/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),check=True,stdout=log,stderr=log)
 paths=[Path(__file__),original,D/'normal-bound.json',d/'lib.rs',d/'candidate.rs',d/'baseline.rs'];r=dict(module=sha(d/'diagnostic.wasm'),command=cmd,explicit_env=old['explicit_env'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in paths},dependency_hashes=old['dependency_hashes'],scope=__doc__);(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(r['module'])
if __name__=='__main__':main()
