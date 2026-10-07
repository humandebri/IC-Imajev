#!/usr/bin/env python3
"""Record saved Rust stage profiles after each fixed-input worker update."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'scripts/measure_update_instruction_profile.py'
    source=p.read_text().replace('artifacts/update-instruction-profile-v1','artifacts/update-other-profile-v1')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
