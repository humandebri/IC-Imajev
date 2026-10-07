#!/usr/bin/env python3
"""Build canonical paid API with exact stack stores for all integer output widths."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_paid_quantize_cached_v2.py';s=p.read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-stack-store-all-v1').replace('artifacts/paid-quantize-cached-v2','artifacts/paid-stack-store-all-v1')
 patch=" source=source.replace(\"artifacts/update-stack-store-all-v1/columns512.wat'][i]\",\"artifacts/update-stack-store-all-v1/columns512.wat','artifacts/update-stack-store-all-v1/int8_168.wat'][i]\").replace(\"artifacts/s1-pair-bounds-v1/build/full-kernel.wat\",\"artifacts/update-stack-store-all-v1/int8_128.wat\").replace(\"if i==9\",\"if i==len(base['patches'])-1\").replace(\"})==10\",\"})==11\")\n (d/'frozen-builder.py').write_text(source)\n"
 anchor=' files=[Path(__file__)';assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
