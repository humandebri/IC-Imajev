#!/usr/bin/env python3
"""Compare four-byte SHA updates with a single initialized float byte view."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    d=ROOT/'artifacts/float-hash-v1/build'
    d.mkdir(parents=True,exist_ok=False)
    source='''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};
use std::cell::Cell;
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{
 OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=1&&!input.is_empty()&&input.len()%4==0&&input.len()<=1_900_000);
 let begin=ic_cdk::api::performance_counter(0);
 let x:Vec<f32>=input.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
 let prepared=ic_cdk::api::performance_counter(0);
 let digest=if method==0 {let mut h=Sha256::new();for v in &x{h.update(v.to_le_bytes());}h.finalize().to_vec()}
 else{assert!(cfg!(target_endian="little"));let bytes=unsafe{std::slice::from_raw_parts(x.as_ptr().cast::<u8>(),x.len()*4)};Sha256::digest(bytes).to_vec()};
 let end=ic_cdk::api::performance_counter(0);
 Measurement{digest,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:x.len()as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
}
'''
    (d/'lib.rs').write_text(source)
    baseline=json.loads((ROOT/'artifacts/f32-stack512-v1/build/report.json').read_text())
    command=baseline['command'][:]
    command[command.index('--crate-name')+1]='imajev_float_hash_probe'
    command[command.index('--edition=2021')+1]=str(d/'lib.rs')
    command[command.index('-o')+1]=str(d/'diagnostic.wasm')
    with (d/'compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**baseline['explicit_env']),check=True,stdout=log,stderr=log)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    dependencies=[Path(command[i+1].split('=',1)[1]) for i,v in enumerate(command) if v=='--extern']
    result=dict(module=sha(d/'diagnostic.wasm'),command=command,explicit_env=baseline['explicit_env'],
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [d/'lib.rs',Path(__file__)]},
                dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies})
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(module=result['module'])))


if __name__=='__main__':main()
