#!/usr/bin/env python3
"""Keep canonical paid accounting and owned scheduler with ordered key-lane scores."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'scripts/build_paid_owned_profile_off.py'
    source=p.read_text().replace('artifacts/paid-owned-profile-off-v1','artifacts/paid-gated-norm-finalize-v1').replace('artifacts/update-profile-off-v1','artifacts/update-gated-norm-finalize-v1')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
