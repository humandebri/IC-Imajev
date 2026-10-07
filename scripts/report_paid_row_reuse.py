#!/usr/bin/env python3
"""Audit all saved paid row-sharing replies and all32 hidden references."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-row-reuse-v1').replace('artifacts/update-finite-max-v1','artifacts/update-row-reuse-v1')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="dict(changed_runtime_files=['delta_head_continue.rs'],added_runtime_files=['delta_head_continue/projection_row_reuse.rs'],arithmetic_kernels_equal=True,scope='Only first-layer capture shares identical Q lanes/scales/QA/ZA rows; scatter before conv, recurrence, state and gating. Restore has no reuse and wire format unchanged.')"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Actual paid bound-Delta path does not call changed delta_head_continue; no sharing is applied. Reported counters belong to this unsuccessful candidate.')
 s=s.replace('summary.update(all_32_hidden_verified=True,','summary.update(optimization_adopted=False,projection_reuse_in_actual_paid_path=False,all_32_hidden_verified=True,')
 s=s.replace('extras=[Path(__file__),p,','extras=[Path(__file__),p,ROOT/\'scripts/report_paid_finite_max.py\',d/\'frozen-row-reporter.py\',')
 (ROOT/'artifacts/paid-row-reuse-v1/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
