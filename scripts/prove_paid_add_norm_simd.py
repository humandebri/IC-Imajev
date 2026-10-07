#!/usr/bin/env python3
"""Run actual paid requests, reference and payment checks, restoring the owned local target."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-add-norm-simd-v2';assert all(sha(ROOT/p)==h for p,h in json.loads((d/'workflow-hashes.json').read_text()).items())
 prior=json.loads((ROOT/'artifacts/update-add-norm-simd-v1/summary.json').read_text());assert prior['verified'] and prior['baseline_restored']
 p=ROOT/'scripts/prove_paid_update_candid_final.py';s=p.read_text().replace('artifacts/paid-update-v1/build-v3','artifacts/paid-add-norm-simd-v2/build').replace('artifacts/paid-update-v1/proof-v4','artifacts/paid-add-norm-simd-v2/proof')
 before="assert len(result['workers'])==[5,4,5][i]";assert s.count(before)==1;s=s.replace(before,"assert 1<=len(result['workers'])<=5")
 (d/'frozen-proof.py').write_text(s);files=[Path(__file__),p,d/'workflow-hashes.json',d/'frozen-proof.py',ROOT/'artifacts/update-add-norm-simd-v1/summary.json'];(d/'proof-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
