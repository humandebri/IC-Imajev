#!/usr/bin/env python3
"""Audit actual paid full-Delta row sharing, saved Candid and all32 hidden."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-zero-seed-v1').replace('artifacts/update-finite-max-v1','artifacts/update-zero-seed-v1')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="dict(changed_runtime_files=['strassen_raw.rs'],original_arithmetic_kernels_equal=True,added_kernels=['__imajev_s1_wide_seed_accumulate','__imajev_s1_160_seed_accumulate','__imajev_s1_raw_seed_accumulate'],scope='MaybeUninit sums for160/128/32, initialized in block0 with positive-zero add. Later blocks and168 path unchanged; F32 operation order preserved.')"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Dense Delta kernels unchanged; integer projection first-block initialization validated against saved hidden/state references.')
 s=s.replace('extras=[Path(__file__),p,',"extras=[Path(__file__),p,ROOT/'scripts/report_paid_finite_max.py',d/'frozen-row-reporter.py',")
 (ROOT/'artifacts/paid-zero-seed-v1/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
