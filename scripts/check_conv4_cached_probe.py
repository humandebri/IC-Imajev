#!/usr/bin/env python3
"""Check the table-borrow variant with the same scalar and ordered F32 oracles."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/check_conv4_probe.py';s=p.read_text().replace('artifacts/conv4-simd-v3','artifacts/conv4-cached-v1')
 (ROOT/'artifacts/conv4-cached-v1/frozen-check.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
