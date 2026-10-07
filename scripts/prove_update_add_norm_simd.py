#!/usr/bin/env python3
"""Verify all fixed proposals with add_norm SIMD and restore the baseline snapshot."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/prove_update_conv4.py';s=p.read_text().replace('artifacts/update-conv4-v1','artifacts/update-add-norm-simd-v1')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
