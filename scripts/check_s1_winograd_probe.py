#!/usr/bin/env python3
"""All21 real/synthetic token shapes, same-module control and independent native oracle."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/check_s1_output160.py';s=p.read_text().replace('artifacts/s1-output160-v1','artifacts/s1-winograd-v1').replace('output160','winograd7').replace('for n in [57,59,67]:','for n in [48,56,57,59,67]:')
 (ROOT/'artifacts/s1-winograd-v1/frozen-check.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
