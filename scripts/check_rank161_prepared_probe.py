#!/usr/bin/env python3
"""Run rank161 versus S1 on saved full-model Q weights and native bits."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT/'scripts/check_s3_stream_probe.py'
    directory = ROOT/'artifacts/rank161-prepared-v1'
    expected = (directory/'check-upstream.sha256').read_text().strip()
    assert hashlib.sha256(upstream.read_bytes()).hexdigest() == expected
    source = upstream.read_text().replace('artifacts/s3-stream-v1','artifacts/rank161-prepared-v1').replace('s3_stream','rank161_prepared')
    assert (directory/'frozen-check.py').read_text() == source
    exec(compile(source,str(directory/'frozen-check.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__ == '__main__':
    main()
