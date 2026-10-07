#!/usr/bin/env python3
"""Measure four-key scores, including transpose cost, against frozen ordered-dot SIMD."""
from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    d=ROOT/'artifacts/attention-key-lanes-v1/build';d.mkdir(parents=True,exist_ok=False)
    old=ROOT/'artifacts/update-profile-off-v1/build/runtime/attention_views.rs'
    text=old.read_text();begin=text.index('fn dot(');end=text.index('#[cfg(test)]',begin)
    control=text[begin:end]
    control+='''
pub fn scores(q:&[f32],k:&[f32],n:usize,width:usize,prefix:usize)->Vec<f32>{
 assert!(width>0&&q.len()==n*width&&k.len()==(n+prefix)*width);
 let mut out=Vec::new();let divisor=(width as f32).sqrt();
 for t in 0..n{for key in k[..(prefix+t+1)*width].chunks_exact(width){out.push(crate::bf(dot(&q[t*width..(t+1)*width],key)/divisor));}}
 out
}
'''
    (d/'control.rs').write_text(control)
    helper=ROOT/'scripts/attention_key_lanes.rs';(d/'candidate.rs').write_bytes(helper.read_bytes())
    source='''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use sha2::{Digest,Sha256};use std::cell::Cell;
mod control;mod candidate;
fn bf(v:f32)->f32{let b=v.to_bits();f32::from_bits(b.wrapping_add(0x7fff+((b>>16)&1))&0xffff0000)}
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{
 OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=1&&input.len()>=12&&input.len()%4==0&&input.len()<=1_900_000);
 let begin=ic_cdk::api::performance_counter(0);
 let u=|at:usize|u32::from_le_bytes(input[at..at+4].try_into().unwrap())as usize;
 let(n,width,prefix)=(u(0),u(4),u(8));assert!(n>0&&n+prefix<=132&&width>0&&width<=256);
 let x:Vec<f32>=input[12..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();assert!(x.len()==(2*n+prefix)*width&&x.iter().all(|v|v.is_finite()));
 let(q,k)=x.split_at(n*width);let prepared=ic_cdk::api::performance_counter(0);
 let out=if method==0{control::scores(q,k,n,width,prefix)}else{candidate::scores(q,k,n,width,prefix)};
 let end=ic_cdk::api::performance_counter(0);
 let raw=unsafe{std::slice::from_raw_parts(out.as_ptr().cast::<u8>(),out.len()*4)};let digest=Sha256::digest(raw).to_vec();
 Measurement{digest,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:out.len()as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
}
'''
    (d/'lib.rs').write_text(source)
    baseline=json.loads((ROOT/'artifacts/float-hash-v1/build/report.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(sha(ROOT/p)==h for p,h in baseline['dependency_hashes'].items())
    cmd=baseline['command'][:];cmd[cmd.index('--crate-name')+1]='imajev_attention_key_lanes_probe';cmd[cmd.index('--edition=2021')+1]=str(d/'lib.rs');cmd[cmd.index('-o')+1]=str(d/'diagnostic.wasm')
    with(d/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**baseline['explicit_env']),check=True,stdout=log,stderr=log)
    files=[Path(__file__),helper,old,d/'lib.rs',d/'control.rs',d/'candidate.rs']
    report=dict(module=sha(d/'diagnostic.wasm'),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes=baseline['dependency_hashes'],command=cmd,explicit_env=baseline['explicit_env'],scope=__doc__,transpose_cost_in_body=True)
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['module'])


if __name__=='__main__':main()
