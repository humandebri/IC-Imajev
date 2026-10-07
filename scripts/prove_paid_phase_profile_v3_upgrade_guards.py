#!/usr/bin/env python3
"""Apply snapshot-protected guard workflow to the owned coarse diagnostic module."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT/'scripts/prove_paid_owned_profile_off_recovery_upgrade_guards.py'
    source = p.read_text().replace('artifacts/paid-owned-profile-off-recovery-v1','artifacts/paid-phase-profile-v3')
    d = ROOT/'artifacts/paid-phase-profile-v3'
    assert json.loads((d/'source-audit.json').read_text())['complete']
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__ == '__main__':
    main()
