#!/usr/bin/env python3
"""Measure the noninstrumented hash candidate with both fixed prefix banks."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'scripts/prove_update_other_profile_reuse.py'
    source=p.read_text().replace('artifacts/update-other-profile-v3','artifacts/update-prefix-hash-v2')
    source=source.replace('scripts/measure_update_other_profile_reuse.py','scripts/measure_update_templates.py')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
