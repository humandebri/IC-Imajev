#!/usr/bin/env python3
"""Verify the restricted adaptive168 candidate on the owned local paid canister."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_paid_quantize_cached_v2.py';s=p.read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-stack-store-v1').replace('artifacts/paid-quantize-cached-v2','artifacts/paid-stack-store-v1')
 patch=" source=source.replace(\"artifacts/update-stack-store-v1/columns512.wat'][i]\",\"artifacts/update-stack-store-v1/columns512.wat','artifacts/update-stack-store-v1/int8_168.wat'][i]\").replace(\"if i==9\",\"if i==len(base['patches'])-1\").replace(\"})==10\",\"})==11\")\n"
 anchor2=' files=[Path(__file__)';assert s.count(anchor2)==1;s=s.replace(anchor2,patch+anchor2)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
