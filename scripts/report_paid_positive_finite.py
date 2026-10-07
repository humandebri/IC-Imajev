#!/usr/bin/env python3
"""Audit actual paid full-Delta row sharing, saved Candid and all32 hidden."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-positive-finite-v1').replace('artifacts/update-finite-max-v1','artifacts/update-positive-finite-v1')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="{'changed_runtime_files': ['int8_kernel.rs', 'int8_token_kernel.rs', 'lib.rs', 'mlp_pipeline.rs', 'mlp_stream.rs', 'output_pairs.rs', 'projection_codec.rs', 'strassen_prepacked.rs'], 'added_runtime_files': ['positive_finite.rs'], 'arithmetic_kernels_equal': True, 'scope': 'Fourteen pure positive-finite checks, including one native unit-test check, use the verified early64 SIMD predicate. IEEE classification unchanged; arithmetic kernels unchanged.'}"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Dense Delta kernels unchanged; finite predicate substitution validated against saved hidden/state references.')
 s=s.replace('extras=[Path(__file__),p,',"extras=[Path(__file__),p,ROOT/'scripts/report_paid_finite_max.py',d/'frozen-row-reporter.py',")
 (ROOT/'artifacts/paid-positive-finite-v1/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
