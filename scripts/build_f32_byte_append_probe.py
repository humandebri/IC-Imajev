#!/usr/bin/env python3
"""Compare scalar F32 byte appends with exact one-copy Wasm append."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/f32-byte-append-v1').replace('scripts/finite_simd.rs','scripts/f32_byte_append.rs').replace('finite_simd.rs','f32_byte_append.rs').replace('mod finite_simd;','mod f32_byte_append;use sha2::{Digest,Sha256};').replace('method<=3','method<=2')
 a=s.index('let mut result=Vec::new();');b=s.index('let end=ic_cdk::api::performance_counter(0);',a)
 s=s[:a]+'''let prefix=stride%17;let mut result:Vec<u8>=(0..prefix).map(|i|((i*37+11)%256)as u8).collect();result.reserve(x.len()*4);
match method{0=>{for v in &x{result.extend(v.to_le_bytes());}},1=>{for v in &x{result.extend_from_slice(&v.to_le_bytes());}},2=>f32_byte_append::append(&mut result,&x),_=>unreachable!()};
''' +s[b:]
 s=s.replace('Measurement{digest:result,','Measurement{digest:Sha256::digest(&result).to_vec(),')
 d=ROOT/'artifacts/f32-byte-append-v1';d.mkdir(exist_ok=True);s=s.replace('[Path(__file__),helper,',"[Path(__file__),helper,p,d/'frozen-builder.py',")
 (d/'frozen-builder.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',d=d,p=p))
if __name__=='__main__':main()
