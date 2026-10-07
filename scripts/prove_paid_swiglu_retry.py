#!/usr/bin/env python3
"""Repeat paid SwiGLU proof with every competing request on current quote version4."""
from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-swiglu-cached-simd-v2';d.mkdir(exist_ok=False)
 old=ROOT/'artifacts/paid-swiglu-cached-simd-v1'
 for name in ['build','workflow-hashes.json','guard-wrapper-hashes.json','upgrade-guards','upgrade-guard-entry-hashes.json','frozen-upgrade-guards.py','optimized-paid-scheduler.rs']:
  (d/name).symlink_to(old/name,target_is_directory=(old/name).is_dir())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'proof-entry-hashes.json').read_text()).items())
 p=old/'frozen-proof.py';s=p.read_text().replace('artifacts/paid-swiglu-cached-simd-v1','artifacts/paid-swiglu-cached-simd-v2')
 before="request_id='busy-secondary',quote_version=2";assert s.count(before)==1;s=s.replace(before,"request_id='busy-secondary',quote_version=4")
 (d/'frozen-proof.py').write_text(s)
 files=[Path(__file__),p,d/'frozen-proof.py',old/'proof-entry-hashes.json',d/'workflow-hashes.json',d/'upgrade-guards/verified.json']
 (d/'proof-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-proof.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
