#!/usr/bin/env python3
"""Compare fused first-use rank7 cache with S1 and independent native bits."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'artifacts/s1-winograd-tail-v1/frozen-check.py'
    d=ROOT/'artifacts/s1-winograd-tail-dispatch-v1'
    source=p.read_text().replace('artifacts/s1-winograd-tail-v1','artifacts/s1-winograd-tail-dispatch-v1')
    (d/'frozen-check.py').write_text(source)
    files=[Path(__file__),p,d/'frozen-check.py',d/'entry-hashes.json']
    (d/'checker-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},indent=2)+'\n')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
