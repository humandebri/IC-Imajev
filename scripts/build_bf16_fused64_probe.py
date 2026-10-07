#!/usr/bin/env python3
"""Compare original checked BF16/F32 expansion with identical unrolled64 loops."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/bf16-fused64-v1').replace('scripts/finite_simd.rs','scripts/bf16_fused64.rs').replace('finite_simd.rs','bf16_fused64.rs').replace('mod finite_simd;','mod bf16_fused64;use sha2::{Digest,Sha256};')
 a=s.index('let prepared=ic_cdk::api::performance_counter(0);');b=s.index('let end=ic_cdk::api::performance_counter(0);',a)
 s=s[:a]+"""let prefix=stride%17;let mut bytes=vec![0u8;prefix];if method<2{bytes.extend(x.iter().flat_map(|v|((v.to_bits()>>16)as u16).to_le_bytes()));}else{bytes.extend_from_slice(&input[4..]);}let mut decoded=vec![0f32;x.len()];let prepared=ic_cdk::api::performance_counter(0);
let outcome=unsafe{match method{0=>bf16_fused64::bf16_8(&bytes[prefix..],decoded.as_mut_ptr(),x.len()).map(|_|1u8),1=>bf16_fused64::bf16_64(&bytes[prefix..],decoded.as_mut_ptr(),x.len()).map(|_|1u8),2=>bf16_fused64::f32_4(&bytes[prefix..],decoded.as_mut_ptr(),x.len()).map(u8::from),3=>bf16_fused64::f32_64(&bytes[prefix..],decoded.as_mut_ptr(),x.len()).map(u8::from),_=>unreachable!()}};
"""+s[b:]
 s=s.replace('Measurement{digest:result,','let status=match outcome{Ok(v)=>v,Err(e)=>{assert_eq!(e,"invalid activation");2u8}};let mut raw:Vec<u8>=decoded.iter().flat_map(|v|v.to_le_bytes()).collect();raw.push(status);Measurement{digest:Sha256::digest(&raw).to_vec(),')
 d=ROOT/'artifacts/bf16-fused64-v1';d.mkdir(exist_ok=True);s=s.replace('[Path(__file__),helper,',"[Path(__file__),helper,p,d/'frozen-builder.py',ROOT/'crates/inference-core/src/bf16.rs',")
 (d/'frozen-builder.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',d=d,p=p))
if __name__=='__main__':main()
