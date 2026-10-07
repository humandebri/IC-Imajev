#!/usr/bin/env python3
"""Re-decode full paid evidence and match deterministic guards to the same module."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_quantize_cached_v2.py';s=p.read_text().replace('artifacts/paid-quantize-cached-v2','artifacts/paid-finite-simd-v1');exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
