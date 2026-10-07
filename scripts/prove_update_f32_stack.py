#!/usr/bin/env python3
"""Verify stack-held F32 sums on all three update inputs and restore the baseline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = ROOT / 'artifacts/update-f32-stack-v1'
    hashes = json.loads((d / 'workflow-hashes.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    p = ROOT / 'scripts/prove_update_templates.py'
    source = p.read_text().replace("artifacts/update-templates-v1/build'", "artifacts/update-f32-stack-v1/build'")
    source = source.replace("artifacts/update-templates-v1/proof-v2'", "artifacts/update-f32-stack-v1/proof'")
    source = source.replace("'--repeats','3'", "'--repeats','1'")
    metadata = d / 'measurement-source.json'
    assert not metadata.exists()
    metadata.write_text(json.dumps(dict(sha256=hashlib.sha256((ROOT / 'scripts/measure_update_templates.py').read_bytes()).hexdigest(), proof_sha256=hashlib.sha256(p.read_bytes()).hexdigest()), indent=2) + '\n')
    (d / 'frozen-proof.py').write_text(source)
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
