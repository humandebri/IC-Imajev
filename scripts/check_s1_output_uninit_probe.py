#!/usr/bin/env python3
"""Compare same-module adaptive160/168 to native exact INT8 outputs."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/check_s1_output160.py';s=p.read_text().replace('artifacts/s1-output160-v1','artifacts/s1-output-uninit-v1')
 s=s.replace('for n in [57,59,67]:','for n in [48,56,57,59,67]:')
 s=s.replace("[(3,'s1_pair_bounds'),(4,'output160')]","[(4,'direct168'),(5,'output_uninit')]")
 s=s.replace("measured['s1_pair_bounds']","measured['direct168']").replace("measured['output160']","measured['output_uninit']")
 d=ROOT/'artifacts/s1-output-uninit-v1';(d/'frozen-check.py').write_text(s);(d/'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
