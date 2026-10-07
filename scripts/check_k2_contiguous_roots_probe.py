#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'artifacts/k2-contiguous-roots-probe-v1/frozen-checker.py'
exec(compile(p.read_text(),str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
