#!/usr/bin/env python3
"""Build or verify exact BF16 pack/unpack overlay on the local paid candidate."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-bf16-pack64-v1').replace('artifacts/update-finite-max-v1','artifacts/update-bf16-pack64-v1')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="{'changed_runtime_files': [], 'added_runtime_files': [], 'changed_dependency': 'inference_core BF16 pack/unpack SIMD bodies and helper module', 'arithmetic_kernels_equal': True, 'scope': 'Runtime source bytes unchanged. Independent core overlay preserves linear/block256/native/public validation bytes; only Wasm pack/unpack body dispatches to independently verified shuffle64/expand64 helper. Original22 arithmetic WAT kernels retained.'}"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Dense Delta kernels unchanged; BF16 pack/unpack overlay validated against saved hidden/state references.')
 s=s.replace('extras=[Path(__file__),p,',"extras=[Path(__file__),p,ROOT/'scripts/report_paid_finite_max.py',d/'frozen-row-reporter.py',")
 (ROOT/'artifacts/paid-bf16-pack64-v1/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
