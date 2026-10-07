#!/usr/bin/env python3
"""Same-module scalar/SIMD BF16 LoRA finalization diagnostic."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/lora-finish-v1/build';d.mkdir(parents=True,exist_ok=False)
 helper=r'''#[inline(always)]fn bf(v:f32)->f32{let b=v.to_bits();f32::from_bits(b.wrapping_add(0x7fff+((b>>16)&1))&0xffff0000)}
#[target_feature(enable="simd128")]
unsafe fn finish(base:&mut[f32],z:&[f32],scale:f32){
 use core::arch::wasm32::*;
 #[inline(always)]unsafe fn round(v:v128)->v128{
  let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));
  v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))
 }
 let mut i=0;let s=f32x4_splat(scale);
 while i+4<=base.len(){
  let b=round(v128_load(base.as_ptr().add(i).cast()));
  let l=round(f32x4_mul(s,v128_load(z.as_ptr().add(i).cast())));
  v128_store(base.as_mut_ptr().add(i).cast(),round(f32x4_add(b,l)));i+=4;
 }
 while i<base.len(){base[i]=bf(bf(base[i])+bf(scale*z[i]));i+=1;}
}
'''
 (d/'finish.rs').write_text(helper)
 source=r'''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use std::cell::Cell;
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{
 OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=1&&input.len()>=12&&(input.len()-4)%8==0&&input.len()<=1_900_000);
 let begin=ic_cdk::api::performance_counter(0);
 let x:Vec<f32>=input.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();let n=(x.len()-1)/2;let scale=x[0];let mut b=x[1..1+n].to_vec();let z=&x[1+n..];
 let prepared=ic_cdk::api::performance_counter(0);
 if method==0 {b=b.into_iter().zip(z).map(|(v,z)|bf(bf(v)+bf(scale*z))).collect();}else{unsafe{finish(&mut b,z,scale)}}
 let end=ic_cdk::api::performance_counter(0);
 let digest=b.iter().flat_map(|v|v.to_le_bytes()).collect();
 Measurement{digest,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:n as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
}
'''+helper
 (d/'lib.rs').write_text(source)
 baseline=json.loads((ROOT/'artifacts/float-hash-v1/build/report.json').read_text());command=baseline['command'][:]
 command[command.index('--crate-name')+1]='imajev_lora_finish_probe';command[command.index('--edition=2021')+1]=str(d/'lib.rs');command[command.index('-o')+1]=str(d/'diagnostic.wasm')
 with (d/'compiler.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=dict(os.environ,**baseline['explicit_env']),check=True,stdout=log,stderr=log)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 deps=[Path(command[i+1].split('=',1)[1]) for i,v in enumerate(command) if v=='--extern']
 result=dict(module=sha(d/'diagnostic.wasm'),command=command,explicit_env=baseline['explicit_env'],source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [d/'lib.rs',d/'finish.rs',Path(__file__)]},dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in deps})
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(result['module'])
if __name__=='__main__':main()
