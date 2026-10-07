#!/usr/bin/env python3
"""Same precomputed heap inference with separately measured bounded stable reads."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s2-k2-prepacked-probe-v1';d=ROOT/'artifacts/s2-k2-prepacked-stable-probe-v1';d.mkdir(exist_ok=False)
 s=(old/'frozen-builder.py').read_text().replace('s2-k2-prepacked-probe-v1','s2-k2-prepacked-stable-probe-v1')
 helper='''
impl Prepared{
 pub fn initialize_stable(&self){
  let bytes=self.bytes();let pages=bytes.div_ceil(65536)as u64;let old=ic_cdk::api::stable_size();
  if pages>old{assert_ne!(ic_cdk::api::stable_grow(pages-old),u64::MAX);}
  let data=unsafe{core::slice::from_raw_parts(self.data.as_ptr().cast::<u8>(),bytes)};
  ic_cdk::api::stable_write(0,data);
 }
}
'''
 query='''
#[ic_cdk::query]
fn coefficient_read_cost(bytes:u32)->Measurement{FIXED.with(|f|{
 let f=f.borrow();let f=f.as_ref().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(f.sealed);let count=bytes as usize;assert!(count<=f.win.as_ref().unwrap().bytes());
 let mut data=vec![0u8;count];let begin=ic_cdk::api::performance_counter(0);ic_cdk::api::stable_read(0,&mut data);let instructions=ic_cdk::api::performance_counter(0)-begin;
 Measurement{digest:Sha256::digest(&data).to_vec(),quantize_instructions:0,input_prepare_instructions:0,project_instructions:instructions,total_instructions:instructions,output_values:count as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
})}
'''
 patch=" p=src/'s2.rs';p.write_text(p.read_text()+"+repr(helper)+")\n p=src/'lib.rs';text=p.read_text();before='f.sealed=true;';assert text.count(before)==1;text=text.replace(before,'f.win.as_ref().unwrap().initialize_stable();'+before);p.write_text(text+"+repr(query)+")\n"
 anchor=" cmd=nb['command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 (d/'frozen-builder.py').write_text(s);files=[Path(__file__),old/'frozen-builder.py',old/'build/report.json',d/'frozen-builder.py'];(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
