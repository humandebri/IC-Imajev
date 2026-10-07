#!/usr/bin/env python3
"""Compare original BF16 pack/unpack with shuffle64 and unpack64."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/bf16-pack64-v1').replace('scripts/finite_simd.rs','scripts/bf16_pack64.rs').replace('finite_simd.rs','bf16_pack64.rs').replace('mod finite_simd;','mod bf16_pack64;use sha2::{Digest,Sha256};')
 a=s.index('let prepared=ic_cdk::api::performance_counter(0);');b=s.index('let end=ic_cdk::api::performance_counter(0);',a)
 s=s[:a]+"""let prefix=stride%17;let header:Vec<u8>=(0..prefix).map(|i|((i*37+11)%256)as u8).collect();let mut bytes=header.clone();bytes.extend(x.iter().flat_map(|v|((v.to_bits()>>16)as u16).to_le_bytes()));let mut output=header;output.resize(prefix+x.len()*2,0);let mut decoded=vec![0f32;x.len()];let prepared=ic_cdk::api::performance_counter(0);
unsafe{match method{0=>bf16_pack64::pack8(&x,&mut output[prefix..]),1=>bf16_pack64::pack64(&x,&mut output[prefix..]),2=>bf16_pack64::unpack8(&bytes[prefix..],&mut decoded),3=>bf16_pack64::unpack64(&bytes[prefix..],&mut decoded),_=>unreachable!()}};
"""+s[b:]
 s=s.replace('Measurement{digest:result,','let digest=if method<2{Sha256::digest(&output).to_vec()}else{Sha256::digest(decoded.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>()).to_vec()};Measurement{digest,')
 d=ROOT/'artifacts/bf16-pack64-v1';d.mkdir(exist_ok=True);s=s.replace('[Path(__file__),helper,',"[Path(__file__),helper,p,d/'frozen-builder.py',ROOT/'crates/inference-core/src/bf16.rs',")
 (d/'frozen-builder.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',d=d,p=p))
if __name__=='__main__':main()
