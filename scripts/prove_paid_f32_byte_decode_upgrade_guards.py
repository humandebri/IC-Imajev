#!/usr/bin/env python3
"""Build or verify exact F32 carry byte append on the local paid candidate."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/prove_paid_quantize_upgrade_guards_v2.py';s=p.read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-f32-byte-decode-v1').replace('artifacts/paid-quantize-cached-v2','artifacts/paid-f32-byte-decode-v1')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
