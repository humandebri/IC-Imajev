#!/usr/bin/env python3
"""Audit actual paid full-Delta row sharing, saved Candid and all32 hidden."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-row-reuse-v2').replace('artifacts/update-finite-max-v1','artifacts/update-row-reuse-v2')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="dict(changed_runtime_files=['delta_full_log.rs','lib.rs'],added_runtime_files=['projection_row_reuse.rs'],arithmetic_kernels_equal=True,scope='Actual paid scheduler uses bound DeltaHybrid -> delta_full_log::evaluate_from_state. First-layer QKV/Z projection sharing only; scatter before unchanged conv/gates/recurrence/state. No changes to delta_head_continue.')"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Dense Delta kernels unchanged; projection row grouping is validated against saved hidden/state references.')
 s=s.replace('extras=[Path(__file__),p,',"extras=[Path(__file__),p,ROOT/'scripts/report_paid_finite_max.py',d/'frozen-row-reporter.py',")
 (ROOT/'artifacts/paid-row-reuse-v2/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
