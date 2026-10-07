#!/usr/bin/env python3
"""Run snapshot-protected upgrade guards for the ordered key-lane candidate."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]


def main():
    d=ROOT/'artifacts/paid-attention-key-lanes-v1'
    assert json.loads((d/'source-audit.json').read_text())['complete']
    p=ROOT/'scripts/prove_paid_owned_profile_off_recovery_upgrade_guards.py'
    source=p.read_text().replace('artifacts/paid-owned-profile-off-recovery-v1','artifacts/paid-attention-key-lanes-v1')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
