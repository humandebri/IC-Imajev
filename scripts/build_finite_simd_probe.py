#!/usr/bin/env python3
"""Measure scalar/4/16/64 finite scans including finite and all nonfinite IEEE classes."""
from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/finite-simd-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);helper=ROOT/'scripts/finite_simd.rs';(D/'finite_simd.rs').write_bytes(helper.read_bytes())
 source='''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use std::cell::Cell;
mod finite_simd;
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement {
OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=3&&input.len()>=4&&input.len()<=1_900_000&&input.len()%4==0);
let begin=ic_cdk::api::performance_counter(0);let stride=u32::from_le_bytes(input[..4].try_into().unwrap())as usize;assert!(stride>0);
let x:Vec<f32>=input[4..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let prepared=ic_cdk::api::performance_counter(0);
let mut result=Vec::new();if x.is_empty(){result.push(1);}else{for chunk in x.chunks(stride){let finite=match method{0=>chunk.iter().all(|v|v.is_finite()),1=>unsafe{finite_simd::scan::<4>(chunk)},2=>finite_simd::all_finite(chunk),3=>unsafe{finite_simd::scan::<64>(chunk)},_=>unreachable!()};result.push(u8::from(finite));}}
let end=ic_cdk::api::performance_counter(0);
Measurement{digest:result,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:x.len()as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
}
'''
 (D/'lib.rs').write_text(source);old=json.loads((ROOT/'artifacts/float-hash-v1/build/report.json').read_text());cmd=old['command'][:]
 cmd[cmd.index('--crate-name')+1]='imajev_finite_simd_probe';cmd[cmd.index('--edition=2021')+1]=str(D/'lib.rs');cmd[cmd.index('-o')+1]=str(D/'diagnostic.wasm')
 assert all(sha(ROOT/p)==h for p,h in old['dependency_hashes'].items())
 with(D/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),stdout=log,stderr=log,check=True)
 r=dict(module=sha(D/'diagnostic.wasm'),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),helper,D/'lib.rs',D/'finite_simd.rs']},dependency_hashes=old['dependency_hashes'],command=cmd,explicit_env=old['explicit_env'],scope=__doc__)
 (D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(r['module'])
if __name__=='__main__':main()
