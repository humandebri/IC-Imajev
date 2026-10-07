#!/usr/bin/env python3
"""Verify finite max scans for every BF16 pattern and arbitrary IEEE F32 bits."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/finite-max-v1';p=ROOT/'scripts/check_finite_simd_probe.py'
 s=p.read_text().replace('artifacts/finite-simd-v1','artifacts/finite-max-v1').replace('range(4):','range(6):').replace('queries=len(result)*4','queries=len(result)*6')
 (d/'frozen-check.py').write_text(s);(d/'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
