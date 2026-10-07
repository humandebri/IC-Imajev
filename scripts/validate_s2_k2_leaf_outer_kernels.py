#!/usr/bin/env python3
"""Freeze matching nine-i32 stubs and validate the four leaf-outer WAT bodies."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/validate_s2_k2_kernels.py';d=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1';s=old.read_text().replace('s2-k2-kernels-v1','s2-k2-leaf-outer-kernels-v1');p=d/'frozen-validator.py';assert not p.exists();p.write_text(s)
 (d/'validator-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n');exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
