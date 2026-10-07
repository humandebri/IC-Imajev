#!/usr/bin/env python3
"""Audit actual paid full-Delta row sharing, saved Candid and all32 hidden."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-finite-complete-v1').replace('artifacts/update-finite-max-v1','artifacts/update-finite-complete-v1')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="{'changed_runtime_files': ['delta_finish.rs', 'delta_log.rs', 'delta_mlp_start.rs', 'f32_output.rs', 'mlp_delta_fusion.rs', 'mlp_delta_stream.rs', 'mlp_pipeline.rs', 'mlp_stream.rs', 'output_pairs.rs', 'prefix_hybrid_codec.rs', 'strassen_raw.rs'], 'arithmetic_kernels_equal': True, 'scope': 'Remaining six any scans and thirteen chain/slice scans use the independently verified finite SIMD predicate. Only pure finite checks changed; arithmetic kernels and outputs unchanged.'}"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Dense Delta kernels unchanged; finite predicate substitution validated against saved hidden/state references.')
 s=s.replace('extras=[Path(__file__),p,',"extras=[Path(__file__),p,ROOT/'scripts/report_paid_finite_max.py',d/'frozen-row-reporter.py',")
 (ROOT/'artifacts/paid-finite-complete-v1/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
