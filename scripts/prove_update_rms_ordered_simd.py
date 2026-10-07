#!/usr/bin/env python3
"""Run all fixed proposals with ordered RMS SIMD then restore the owned local target."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/prove_update_conv4.py';s=p.read_text().replace('artifacts/update-conv4-v1','artifacts/update-rms-ordered-simd-v1')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
