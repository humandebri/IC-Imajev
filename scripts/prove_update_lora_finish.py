#!/usr/bin/env python3
"""Compare the arithmetic/hash candidate on three frozen proposals then restore."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/prove_update_prefix_hash_reuse.py';s=p.read_text().replace('artifacts/update-prefix-hash-v2','artifacts/update-lora-finish-v1')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
