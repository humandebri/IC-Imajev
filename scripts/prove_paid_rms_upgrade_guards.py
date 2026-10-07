#!/usr/bin/env python3
"""Exercise deterministic owner-only active/refund fixtures on the latest paid runtime."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-rms-ordered-simd-v1';assert all(sha(ROOT/v)==h for v,h in json.loads((d/'workflow-hashes.json').read_text()).items())
 p=ROOT/'scripts/prove_paid_update_upgrade_guards.py';s=p.read_text().replace('artifacts/paid-update-v1/build-upgrade-diagnostic','artifacts/paid-rms-ordered-simd-v1/build').replace('artifacts/paid-update-v1/upgrade-guards-v1','artifacts/paid-rms-ordered-simd-v1/upgrade-guards')
 (d/'frozen-upgrade-guards.py').write_text(s);paths=[Path(__file__),p,d/'workflow-hashes.json',d/'frozen-upgrade-guards.py'];(d/'upgrade-guard-entry-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v) for v in paths},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
