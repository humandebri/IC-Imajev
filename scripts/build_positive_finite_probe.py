#!/usr/bin/env python3
"""Measure exact positive-finite scans used by quantization scales."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/positive-finite-v1').replace('scripts/finite_simd.rs','scripts/positive_finite.rs').replace('finite_simd.rs','positive_finite.rs').replace('mod finite_simd;','mod positive_finite;').replace('method<=3','method<=2')
 old='0=>chunk.iter().all(|v|v.is_finite()),1=>unsafe{finite_simd::scan::<4>(chunk)},2=>finite_simd::all_finite(chunk),3=>unsafe{finite_simd::scan::<64>(chunk)},_=>unreachable!()'
 assert s.count(old)==1;s=s.replace(old,'0=>chunk.iter().all(|v|v.is_finite()&&*v>0.),1=>unsafe{positive_finite::scan_max::<true>(chunk)},2=>positive_finite::all_positive_finite(chunk),_=>unreachable!()')
 d=ROOT/'artifacts/positive-finite-v1';d.mkdir(exist_ok=True)
 s=s.replace('[Path(__file__),helper,',"[Path(__file__),helper,p,d/'frozen-builder.py',")
 (d/'frozen-builder.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',d=d,p=p))
if __name__=='__main__':main()
