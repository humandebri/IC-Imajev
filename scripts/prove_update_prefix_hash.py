#!/usr/bin/env python3
"""Verify all three fixed inputs with bound prefixes and large-slice bulk hashes."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'scripts/prove_update_f32_shapes.py'
    source=p.read_text().replace('artifacts/update-f32-shapes-v1','artifacts/update-prefix-hash-v1')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
