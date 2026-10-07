#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'artifacts/s2-cached-tail-dot-reuse-probe-v1/frozen-mlp-checker.py'
exec(compile(p.read_text(),str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
