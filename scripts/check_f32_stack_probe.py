#!/usr/bin/env python3
"""Check stack-held F32 accumulation against native scalar and output32 control."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/check_f32_output128_probe.py'
    source = p.read_text().replace('artifacts/f32-output128-v1', 'artifacts/f32-stack-v1')
    d = ROOT / 'artifacts/f32-stack-v1'
    (d / 'frozen-check.py').write_text(source)
    (d / 'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '\n')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
