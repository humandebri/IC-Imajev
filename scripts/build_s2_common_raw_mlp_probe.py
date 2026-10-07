#!/usr/bin/env python3
"""Shared-bank rank49 diagnostic with bounded staged F32 input for true MLP shapes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s2-common-raw-probe-v1/frozen-builder.py';d=ROOT/'artifacts/s2-common-raw-mlp-probe-v1';d.mkdir(exist_ok=False)
 s=original.read_text().replace('s2-common-raw-probe-v1','s2-common-raw-mlp-probe-v1').replace('src.mkdir(parents=True,exist_ok=True)','src.mkdir(parents=True)')
 before=" (src/'lib.rs').write_text(lib)";assert s.count(before)==1
 insertion=''' lib=lib.replace('sealed:bool}', 'sealed:bool,input:Vec<u8>}').replace('win:None,sealed:false}', 'win:None,sealed:false,input:vec![]}')
 lib=lib.replace('let n=input.len()/(f.cols*4);', 'let input=if input.is_empty(){f.input.as_slice()}else{input.as_slice()};let n=input.len()/(f.cols*4);')
 lib=lib.replace('f.sealed && !input.is_empty() && input.len()<=1_500_000 && input.len()%(f.cols*4)==0','f.sealed && (input.is_empty()||input.len()<=1_500_000) && input.len()%(f.cols*4)==0')
 lib=lib.replace('let n=input.len()/(f.cols*4);', 'assert!(!input.is_empty()&&input.len()<=132*f.cols*4&&input.len()%(f.cols*4)==0);let n=input.len()/(f.cols*4);')
 lib+='\\n#[ic_cdk::update]fn stage_input(start:u32,bytes:Vec<u8>)->Preparation{FIXED.with(|s|{let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(f.sealed&&!bytes.is_empty()&&bytes.len()<=1_500_000&&start as usize+bytes.len()<=132*f.cols*4);let begin=ic_cdk::api::performance_counter(0);if start==0{f.input.clear();}assert_eq!(start as usize,f.input.len());f.input.extend_from_slice(&bytes);Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:bytes.len()as u64,rows:0}})}\\n'
'''
 s=s.replace(before,insertion+before)
 frozen=d/'frozen-builder.py';frozen.write_text(s);(d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
