#!/usr/bin/env python3
"""Compare original SIMD4 BF16 predicates to equal-summary SIMD64 loops."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/bf16-predicates-v1').replace('scripts/finite_simd.rs','scripts/bf16_predicates.rs').replace('finite_simd.rs','bf16_predicates.rs').replace('mod finite_simd;','mod bf16_predicates;')
 before='let finite=match method{0=>chunk.iter().all(|v|v.is_finite()),1=>unsafe{finite_simd::scan::<4>(chunk)},2=>finite_simd::all_finite(chunk),3=>unsafe{finite_simd::scan::<64>(chunk)},_=>unreachable!()};result.push(u8::from(finite));'
 after='let value=match method{0=>u8::from(unsafe{bf16_predicates::all4(chunk)}),1=>u8::from(unsafe{bf16_predicates::all64(chunk)}),2=>match unsafe{bf16_predicates::classify4(chunk)}{Ok(b)=>u8::from(b),Err(e)=>{assert_eq!(e,"invalid activation");2}},3=>match unsafe{bf16_predicates::classify64(chunk)}{Ok(b)=>u8::from(b),Err(e)=>{assert_eq!(e,"invalid activation");2}},_=>unreachable!()};result.push(value);'
 assert s.count(before)==1;s=s.replace(before,after)
 d=ROOT/'artifacts/bf16-predicates-v1';d.mkdir(exist_ok=True);s=s.replace('[Path(__file__),helper,',"[Path(__file__),helper,p,d/'frozen-builder.py',ROOT/'crates/inference-core/src/bf16.rs',")
 (d/'frozen-builder.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',d=d,p=p))
if __name__=='__main__':main()
