#!/usr/bin/env python3
"""Verify paid upgrade guards on the newly reconstructed local goal canister."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/paid-owned-profile-off-recovery-v1';ready=json.loads((ROOT/'artifacts/local-goal-recovery-v1/baseline-ready.json').read_text());assert ready['complete'] and ready['target']=='4caro-hl777-77775-aaaba-cai'
 p=ROOT/'scripts/prove_paid_update_upgrade_guards.py';s=p.read_text().replace('artifacts/paid-update-v1/build-upgrade-diagnostic','artifacts/paid-owned-profile-off-recovery-v1/build').replace('artifacts/paid-update-v1/upgrade-guards-v1','artifacts/paid-owned-profile-off-recovery-v1/upgrade-guards').replace('6eydd-o3777-77775-aaama-cai','4caro-hl777-77775-aaaba-cai');(d/'frozen-upgrade-guards.py').write_text(s)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),p,d/'workflow-hashes.json',d/'frozen-upgrade-guards.py',ROOT/'artifacts/local-goal-recovery-v1/baseline-ready.json'];(d/'upgrade-guard-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in paths},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
