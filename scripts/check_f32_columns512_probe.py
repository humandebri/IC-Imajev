#!/usr/bin/env python3
"""Check current stack64 and columns512 against native on actual gate-A input."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/check_f32_output128_probe.py'
    source = p.read_text().replace('artifacts/f32-output128-v1', 'artifacts/f32-columns512-v1')
    source = source.replace("choices=['down-B']", "choices=['gate-A']")
    source = source.replace('rows%128 == 0 and cols%64 == 0', 'rows%64 == 0 and cols%512 == 0')
    source = source.replace("[(0,'output32'),(1,'output128')]", "[(1,'output32'),(2,'output128')]")
    source = source.replace('output32', 'stack64').replace('output128', 'columns512')
    d = ROOT / 'artifacts/f32-columns512-v1'
    (d/'frozen-check.py').write_text(source)
    (d/'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
