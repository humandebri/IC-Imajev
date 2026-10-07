#!/usr/bin/env python3
"""Repeat exact full paid proof on the newly reconstructed local goal canister."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/paid-owned-profile-off-recovery-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();g=json.loads((d/'upgrade-guards/verified.json').read_text());assert g['complete'] and g['baseline_restored'] and g['module']==json.loads((d/'build/report.json').read_text())['wasm_sha256']
 old=ROOT/'artifacts/paid-owned-profile-off-v1';p=old/'frozen-proof.py';manifest=json.loads((old/'proof-entry-hashes.json').read_text());assert all(sha(ROOT/name)==h for name,h in manifest.items())
 s=p.read_text().replace('artifacts/paid-owned-profile-off-v1','artifacts/paid-owned-profile-off-recovery-v1').replace('6eydd-o3777-77775-aaama-cai','4caro-hl777-77775-aaaba-cai');(d/'frozen-proof.py').write_text(s)
 paths=[Path(__file__),p,old/'proof-entry-hashes.json',d/'workflow-hashes.json',d/'frozen-proof.py',d/'upgrade-guards/verified.json',d/'upgrade-guard-entry-hashes.json'];(d/'proof-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in paths},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
