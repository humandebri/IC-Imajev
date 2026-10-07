#!/usr/bin/env python3
"""Compare original checked conv_state with ordered four-channel SIMD."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/conv4-simd-v1/build';d.mkdir(parents=True,exist_ok=False)
 base=ROOT/'artifacts/update-lora-finish-v1/build';r=json.loads((base/'report.json').read_text())
 assert all(sha(ROOT/p)==h for p,h in r['source_hashes'].items())
 shutil.copytree(base/'runtime',d/'runtime')
 helper=r'''#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn conv4_simd(x:&[f32],weight:&[f32],n:usize,c:usize)->Vec<f32>{
 use core::arch::wasm32::*;
 assert!(n>0&&c>0&&c%4==0&&x.len()==(n+3)*c&&weight.len()==c*4);
 let mut y=vec![0.;n*c];
 for ch in (0..c).step_by(4){
  let a=v128_load(weight.as_ptr().add(ch*4).cast());let b=v128_load(weight.as_ptr().add((ch+1)*4).cast());
  let c0=v128_load(weight.as_ptr().add((ch+2)*4).cast());let d=v128_load(weight.as_ptr().add((ch+3)*4).cast());
  let lo0=i32x4_shuffle::<0,4,1,5>(a,b);let lo1=i32x4_shuffle::<0,4,1,5>(c0,d);
  let hi0=i32x4_shuffle::<2,6,3,7>(a,b);let hi1=i32x4_shuffle::<2,6,3,7>(c0,d);
  let w0=i32x4_shuffle::<0,1,4,5>(lo0,lo1);let w1=i32x4_shuffle::<2,3,6,7>(lo0,lo1);
  let w2=i32x4_shuffle::<0,1,4,5>(hi0,hi1);let w3=i32x4_shuffle::<2,3,6,7>(hi0,hi1);
  for t in 0..n{
   let mut sum=f32x4_splat(0.);
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add(t*c+ch).cast()),w0));
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add((t+1)*c+ch).cast()),w1));
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add((t+2)*c+ch).cast()),w2));
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add((t+3)*c+ch).cast()),w3));
   let p=y.as_mut_ptr().add(t*c+ch);
   *p=bf_silu(bf(f32x4_extract_lane::<0>(sum)));*p.add(1)=bf_silu(bf(f32x4_extract_lane::<1>(sum)));
   *p.add(2)=bf_silu(bf(f32x4_extract_lane::<2>(sum)));*p.add(3)=bf_silu(bf(f32x4_extract_lane::<3>(sum)));
  }
 }
 y.extend_from_slice(&x[x.len()-3*c..]);y
}
'''
 (d/'conv4.rs').write_text(helper);p=d/'runtime/lib.rs';p.write_text(p.read_text()+helper)
 cmd=r['runtime_command'][:];cmd[cmd.index('--edition=2021')+1]=str(p);cmd[cmd.index('-o')+1]=str(d/'libimajev_runtime.rlib')
 with (d/'runtime-compiler.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=log)
 source=r'''use candid::{CandidType,Principal};use serde::{Deserialize,Serialize};use std::cell::Cell;use sha2::{Digest,Sha256};
thread_local!{static OWNER:Cell<Principal>=Cell::new(Principal::anonymous());}
#[ic_cdk::init]fn init(owner:Principal,_rows:u32,_cols:u32){assert_ne!(owner,Principal::anonymous());OWNER.with(|o|o.set(owner));imajev_runtime::prepared_activation::prepare();}
#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}
#[derive(CandidType,Deserialize,Serialize)]pub struct Measurement{pub digest:Vec<u8>,pub quantize_instructions:u64,pub input_prepare_instructions:u64,pub project_instructions:u64,pub total_instructions:u64,pub output_values:u64,pub heap_pages:u64}
#[ic_cdk::query]fn project(input:Vec<u8>,method:u8)->Measurement{
 OWNER.with(|o|assert_eq!(o.get(),ic_cdk::api::msg_caller()));assert!(method<=1&&input.len()>=12&&input.len()%4==0&&input.len()<=2_080_000);
 let begin=ic_cdk::api::performance_counter(0);
 let read=|i|u32::from_le_bytes(input[i..i+4].try_into().unwrap()) as usize;let(n,c,k)=(read(0),read(4),read(8));assert!(n>0&&c>0&&k==4&&c%4==0&&input.len()==12+((n+3)*c+c*4)*4);
 let v:Vec<f32>=input[12..].chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();let (x,w)=v.split_at((n+3)*c);
 let r:imajev_runtime::Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"conv_state","tensor":"conv","dims":[n,c,k,0],"scalars":[]})).unwrap();
 let prepared=ic_cdk::api::performance_counter(0);
 let out=if method==0 {imajev_runtime::execute(&r,x,w).unwrap()}else{unsafe{imajev_runtime::conv4_simd(x,w,n,c)}};
 let end=ic_cdk::api::performance_counter(0);let bytes=unsafe{std::slice::from_raw_parts(out.as_ptr().cast::<u8>(),out.len()*4)};let digest=Sha256::digest(bytes).to_vec();
 Measurement{digest,quantize_instructions:0,input_prepare_instructions:prepared-begin,project_instructions:end-prepared,total_instructions:end-begin,output_values:out.len()as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
}
'''
 (d/'lib.rs').write_text(source)
 old=json.loads((ROOT/'artifacts/float-hash-v1/build/report.json').read_text());command=old['command'][:];command[command.index('--crate-name')+1]='imajev_conv4_probe';command[command.index('--edition=2021')+1]=str(d/'lib.rs');command[command.index('-o')+1]=str(d/'diagnostic.wasm')
 command+=['--extern','imajev_runtime='+str(d/'libimajev_runtime.rlib')]
 with (d/'compiler.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),check=True,stdout=log,stderr=log)
 deps=[Path(cmd[i+1].split('=',1)[1]) for i,v in enumerate(cmd) if v=='--extern']+[Path(command[i+1].split('=',1)[1]) for i,v in enumerate(command) if v=='--extern']
 files=[Path(__file__),d/'lib.rs',d/'conv4.rs',d/'libimajev_runtime.rlib']+list((d/'runtime').glob('*.rs'))
 result=dict(module=sha(d/'diagnostic.wasm'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in deps},runtime_command=cmd,command=command,explicit_env=old['explicit_env'],baseline_runtime_report_sha256=sha(base/'report.json'))
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(result['module'])
if __name__=='__main__':main()
