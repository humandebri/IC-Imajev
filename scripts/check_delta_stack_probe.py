#!/usr/bin/env python3
"""Compare stack Delta and current register Delta with native output/state bits."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/check_delta_register_probe.py'
    source = p.read_text().replace('artifacts/delta-register-v3', 'artifacts/delta-stack-v1')
    source = source.replace("'register' if candidate else 'baseline'", "'stack' if candidate else 'register'")
    source = source.replace("measurements['baseline']", "measurements['register']").replace("after=measurements['register']", "after=measurements['stack']")
    d = ROOT / 'artifacts/delta-stack-v1'
    (d / 'frozen-check.py').write_text(source)
    (d / 'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '\n')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
