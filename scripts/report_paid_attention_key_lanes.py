#!/usr/bin/env python3
"""Audit full paid references including independent hidden30 for ordered scores."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'scripts/report_paid_owned_profile_off_recovery.py'
    source=p.read_text().replace('artifacts/paid-owned-profile-off-recovery-v1','artifacts/paid-attention-key-lanes-v1').replace('artifacts/update-profile-off-v1','artifacts/update-attention-key-lanes-v1')
    comparison=json.loads((ROOT/'artifacts/update-attention-key-lanes-v1/runtime-comparison.json').read_text())
    source,count=re.subn(r'^ new=.*$',' new='+repr(repr(comparison)),source,flags=re.M);assert count==1
    source=source.replace("d/'scheduler-source-audit.json'","d/'source-audit.json'")
    source=source.replace('owned scheduler with optional traces compiled out validated against saved hidden/state references.','owned scheduler and ordered key-lane attention scores validated against saved hidden/state references.')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
