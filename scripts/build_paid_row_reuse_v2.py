#!/usr/bin/env python3
"""Build canonical paid API with first-layer exact projection row sharing."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_paid_quantize_cached_v2.py';s=p.read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-row-reuse-v2').replace('artifacts/paid-quantize-cached-v2','artifacts/paid-row-reuse-v2')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
