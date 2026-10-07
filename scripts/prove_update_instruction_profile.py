#!/usr/bin/env python3
"""Run three fixed update inputs with profiling under the existing snapshot guard."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source_path = ROOT/'scripts/prove_update_templates.py'
    source = source_path.read_text()
    source = source.replace("artifacts/update-templates-v1/build'", "artifacts/update-instruction-profile-v1/build-v3'")
    source = source.replace("artifacts/update-templates-v1/proof-v2'", "artifacts/update-instruction-profile-v1/proof-v1'")
    source = source.replace('scripts/measure_update_templates.py', 'scripts/measure_update_instruction_profile.py')
    source = source.replace("'--repeats','3'", "'--repeats','1'")
    d = ROOT/'artifacts/update-instruction-profile-v1'
    metadata = d/'measurement-source.json'
    assert not metadata.exists()
    metadata.write_text(json.dumps(dict(sha256=hashlib.sha256((ROOT/'scripts/measure_update_templates.py').read_bytes()).hexdigest(), proof_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()), indent=2)+'\n')
    (d/'frozen-proof.py').write_text(source)
    exec(compile(source, str(source_path), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
