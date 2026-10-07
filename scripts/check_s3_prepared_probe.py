#!/usr/bin/env python3
"""Check immutable-prepared rank343 against latest S1 and native integers."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/check_s3_stream_probe.py'
    source = p.read_text().replace('artifacts/s3-stream-v1', 'artifacts/s3-prepared-v1')
    source = source.replace('s3_stream', 's3_prepared')
    d = ROOT / 'artifacts/s3-prepared-v1'
    (d / 'frozen-check.py').write_text(source)
    (d / 'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '\n')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
