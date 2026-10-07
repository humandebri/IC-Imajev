#!/usr/bin/env python3
"""Reuse all 19 independent-oracle cases for the shared-transform rank49 candidate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT/'scripts/check_s1_output160.py'
    source = p.read_text()
    source = source.replace('artifacts/s1-output160-v1/build','artifacts/s2-winograd-hoisted-v1/build')
    source = source.replace('artifacts/s1-output160-v1/check','artifacts/s2-winograd-hoisted-v1/check')
    source = source.replace('output160','s2_winograd_hoisted')
    source = source.replace('streamed rank49/output64','factored rank49/output88 with late pair reduction')
    source=source.replace('for n in [57,59,67]:','for n in [48,56,57,59,67]:')
    (ROOT/'artifacts/s2-winograd-hoisted-v1/frozen-check.py').write_text(source)
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__ == '__main__':
    main()
