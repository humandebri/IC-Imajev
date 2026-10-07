#!/usr/bin/env python3
"""Run frozen independent score oracle against the bounded unrolled component."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def main():
    d=ROOT/'artifacts/attention-key-lanes-bounded-v1'
    p=ROOT/'scripts/check_attention_key_lanes_probe.py'
    source=p.read_text().replace('artifacts/attention-key-lanes-v1','artifacts/attention-key-lanes-bounded-v1')
    (d/'frozen-checker.py').write_text(source)
    files=[Path(__file__),p,d/'frozen-checker.py',d/'entry-hashes.json']
    (d/'checker-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},indent=2)+'\n')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
