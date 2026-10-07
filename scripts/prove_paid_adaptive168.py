#!/usr/bin/env python3
"""Verify the restricted adaptive168 candidate on the owned local paid canister."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/prove_paid_quantize_cached_v2.py';s=p.read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-adaptive168-v1').replace('artifacts/paid-quantize-cached-v2','artifacts/paid-adaptive168-v1')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
