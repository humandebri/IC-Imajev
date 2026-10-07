#!/usr/bin/env python3
"""Same-runtime scalar/SIMD SwiGLU with immutable-table borrowing and owned gate reuse."""
from pathlib import Path
import hashlib,json,shutil,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/swiglu-cached-simd-v1/build';d.mkdir(parents=True,exist_ok=False);base=ROOT/'artifacts/update-rms-ordered-simd-v1/build';b=json.loads((base/'report.json').read_text());assert all(sha(ROOT/p)==h for p,h in b['source_hashes'].items());shutil.copytree(base/'runtime',d/'runtime')
 helper=r'''pub fn swiglu_scalar(gate:Vec<f32>,up:Vec<f32>)->Vec<f32>{gate.into_iter().zip(up).map(|(g,u)|bf(bf_silu(g)*u)).collect()}
pub fn swiglu_owned(mut gate:Vec<f32>,up:Vec<f32>)->Vec<f32>{
 assert_eq!(gate.len(),up.len());
 #[cfg(target_arch="wasm32")]
 if gate.len()>=4{prepared_activation::with_table::<1,_>(|table|unsafe{swiglu_simd(&mut gate,&up,table)});return gate;}
 swiglu_scalar(gate,up)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn swiglu_simd(gate:&mut[f32],up:&[f32],table:Option<&[f32;65536]>){
use core::arch::wasm32::*;
#[inline(always)]unsafe fn round(v:v128)->v128{let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))}
let emit=|g:f32|{let bits=g.to_bits();if bits&65535==0 && g.is_finite(){if let Some(t)=table{return t[(bits>>16)as usize];}}bf_silu_original(g)};
let mut i=0;
while i+4<=gate.len(){let g=gate.as_mut_ptr().add(i);let v=f32x4(emit(*g),emit(*g.add(1)),emit(*g.add(2)),emit(*g.add(3)));let y=round(f32x4_mul(v,v128_load(up.as_ptr().add(i).cast())));v128_store(g.cast(),y);i+=4;}
while i<gate.len(){gate[i]=bf(emit(gate[i])*up[i]);i+=1;}
}
'''
 (d/'swiglu.rs').write_text(helper);p=d/'runtime/lib.rs';p.write_text(p.read_text()+helper);cmd=b['runtime_command'][:];cmd[cmd.index('--edition=2021')+1]=str(p);cmd[cmd.index('-o')+1]=str(d/'libimajev_runtime.rlib')
 with (d/'runtime-compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=log)
 source=r'''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use std::cell::Cell;use sha2::{Digest,Sha256};
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));imajev_runtime::prepared_activation::prepare();}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{
 OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=3&&input.len()>=12&&(input.len()-4)%8==0&&input.len()<=1_900_000);
 let begin=ic_cdk::api::performance_counter(0);let n=u32::from_le_bytes(input[..4].try_into().unwrap()) as usize;assert!(n>0&&n<=900000);let seed:Vec<f32>=input[4..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();assert!(seed.iter().all(|v|v.is_finite()));let pairs=seed.len()/2;
 let gate:Vec<_>=(0..n).map(|i|seed[(i%pairs)*2]).collect();let up:Vec<_>=(0..n).map(|i|seed[(i%pairs)*2+1]).collect();if method>=2{imajev_runtime::prepared_activation::clear();}
 let prepared=ic_cdk::api::performance_counter(0);let out=if method%2==0{imajev_runtime::swiglu_scalar(gate,up)}else{imajev_runtime::swiglu_owned(gate,up)};
 let end=ic_cdk::api::performance_counter(0);let bytes=unsafe{std::slice::from_raw_parts(out.as_ptr().cast::<u8>(),out.len()*4)};let digest=Sha256::digest(bytes).to_vec();Measurement{digest,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:out.len() as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
}
'''
 (d/'lib.rs').write_text(source);old=json.loads((ROOT/'artifacts/float-hash-v1/build/report.json').read_text());command=b['command'][:];command[command.index('--crate-name')+1]='imajev_swiglu_probe';command[command.index('--edition=2021')+1]=str(d/'lib.rs');command[command.index('-o')+1]=str(d/'diagnostic.wasm');command=[v if not v.startswith('imajev_runtime=')else 'imajev_runtime='+str(d/'libimajev_runtime.rlib') for v in command]
 with (d/'compiler.log').open('w')as log:subprocess.run(command,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),check=True,stdout=log,stderr=log)
 deps=[Path(c[i+1].split('=',1)[1]) for c in [cmd,command] for i,v in enumerate(c) if v=='--extern'];files=[Path(__file__),base/'report.json',d/'lib.rs',d/'swiglu.rs']+list((d/'runtime').glob('*.rs'))
 r=dict(module=sha(d/'diagnostic.wasm'),runtime_command=cmd,command=command,explicit_env=old['explicit_env'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in deps});(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(r['module'])
if __name__=='__main__':main()
