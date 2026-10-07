#!/usr/bin/env python3
"""Build or verify exact F32 carry byte append on the local paid candidate."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_paid_quantize_cached_v2.py';s=p.read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-bf16-corepred-v1').replace('artifacts/paid-quantize-cached-v2','artifacts/paid-bf16-corepred-v1')
 patch=" source=source.replace(\"artifacts/update-bf16-corepred-v1/columns512.wat'][i]\",\"artifacts/update-bf16-corepred-v1/columns512.wat','artifacts/update-bf16-corepred-v1/int8_168.wat','artifacts/update-bf16-corepred-v1/seed128.wat','artifacts/update-bf16-corepred-v1/seed160.wat','artifacts/update-bf16-corepred-v1/seed32.wat', 'artifacts/update-bf16-corepred-v1/direct128.wat', 'artifacts/update-bf16-corepred-v1/direct128-seed.wat', 'artifacts/update-bf16-corepred-v1/direct160.wat', 'artifacts/update-bf16-corepred-v1/direct160-seed.wat', 'artifacts/update-bf16-corepred-v1/direct32.wat', 'artifacts/update-bf16-corepred-v1/direct32-seed.wat','artifacts/update-bf16-corepred-v1/direct168.wat','artifacts/update-bf16-corepred-v1/direct168-seed.wat'][i]\").replace(\"artifacts/s1-pair-bounds-v1/build/full-kernel.wat\",\"artifacts/update-bf16-corepred-v1/int8_128.wat\").replace(\"if i==9\",\"if i==len(base['patches'])-1\").replace(\"})==10\",\"})==22\")\n (d/'frozen-builder.py').write_text(source)\n"
 anchor=' files=[Path(__file__)';assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
