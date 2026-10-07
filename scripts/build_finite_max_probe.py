#!/usr/bin/env python3
"""Compare exact shifted-bit unsigned max scans with verified finite SIMD64."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/finite-max-v1';d.mkdir(exist_ok=False)
 p=ROOT/'scripts/build_finite_simd_probe.py';s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/finite-max-v1')
 s=s.replace('mod finite_simd;','mod finite_simd;mod finite_max;').replace('method<=3','method<=5')
 old='3=>unsafe{finite_simd::scan::<64>(chunk)},_=>unreachable!()';assert s.count(old)==1
 s=s.replace(old,'3=>unsafe{finite_simd::scan::<64>(chunk)},4=>unsafe{finite_max::scan_max::<true>(chunk)},5=>unsafe{finite_max::scan_max::<false>(chunk)},_=>unreachable!()')
 anchor=" (D/'lib.rs').write_text(source);";assert s.count(anchor)==1
 s=s.replace(anchor," (D/'finite_max.rs').write_bytes((ROOT/'scripts/finite_max.rs').read_bytes())\n"+anchor)
 s=s.replace("helper,D/'lib.rs',D/'finite_simd.rs'","helper,ROOT/'scripts/finite_max.rs',D/'lib.rs',D/'finite_simd.rs',D/'finite_max.rs',d/'frozen-builder.py',p")
 (d/'frozen-builder.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',d=d,p=p))
if __name__=='__main__':main()
