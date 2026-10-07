#!/usr/bin/env python3
"""Compare exact pair-late S2 and current pair-bounds S1 with the native oracle."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT/'scripts/check_s1_output160.py'
    source = p.read_text()
    source = source.replace("artifacts/s1-output160-v1/build'", "artifacts/s2-pair-late-v1/build-v3'")
    source = source.replace("artifacts/s1-output160-v1/check'", "artifacts/s2-pair-late-v1/check-v2'")
    source = source.replace('output160','s2_pair_late')
    source = source.replace('latest S1 pair-bounds and streamed rank49/output64', 'current S1 pair-bounds and rank49 with paired lanes/late reduction')
    d = ROOT/'artifacts/s2-pair-late-v1'
    (d/'check-source.json').write_text(json.dumps(dict(upstream_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),transformed_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2)+'\n')
    (d/'frozen-check.py').write_text(source)
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__ == '__main__':
    main()
