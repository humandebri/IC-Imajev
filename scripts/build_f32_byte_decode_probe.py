#!/usr/bin/env python3
"""Compare scalar decoding with exact one-copy Wasm decode."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/f32-byte-decode-v1').replace('scripts/finite_simd.rs','scripts/f32_byte_decode.rs').replace('finite_simd.rs','f32_byte_decode.rs').replace('mod finite_simd;','mod f32_byte_decode;use sha2::{Digest,Sha256};').replace('method<=3','method<=2')
 a=s.index('let x:Vec<f32>=');b=s.index('let end=ic_cdk::api::performance_counter(0);',a)
 s=s[:a]+"""let prefix=stride%17;let mut bytes=vec![0u8;prefix];bytes.extend_from_slice(&input[4..]);bytes.extend(core::iter::repeat(0xabu8).take(stride%4));let raw=&bytes[prefix..];let prepared=ic_cdk::api::performance_counter(0);
let result:Vec<f32>=match method{0=>raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect(),1=>{let mut out=Vec::with_capacity(raw.len()/4);for b in raw.chunks_exact(4){out.push(f32::from_le_bytes(b.try_into().unwrap()));}out},2=>f32_byte_decode::decode(raw),_=>unreachable!()};
"""+s[b:]
 s=s.replace('Measurement{digest:result,','let digest:Vec<u8>=result.iter().flat_map(|v|v.to_le_bytes()).collect();Measurement{digest:Sha256::digest(&digest).to_vec(),').replace('output_values:x.len()','output_values:result.len()')
 d=ROOT/'artifacts/f32-byte-decode-v1';d.mkdir(exist_ok=True);s=s.replace('[Path(__file__),helper,',"[Path(__file__),helper,p,d/'frozen-builder.py',")
 (d/'frozen-builder.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',d=d,p=p))
if __name__=='__main__':main()
