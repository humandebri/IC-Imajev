#!/usr/bin/env python3
"""Measure adaptive output160/128/32 against current output128 and native INT8."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/check_s1_output160.py'
    source = p.read_text().replace('artifacts/s1-output160-v1', 'artifacts/s1-mul-pairwise-v1')
    source = source.replace('output160', 'mul_pairwise')
    source=source.replace('for n in [57,59,67]:','for n in [48,56,57,59,67]:')
    d = ROOT / 'artifacts/s1-mul-pairwise-v1'
    (d / 'frozen-check.py').write_text(source)
    (d / 'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '\n')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
