#!/usr/bin/env python3
"""Run the existing deterministic guard fixtures after a restored paid proof."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    assert directory.is_relative_to(ROOT / 'artifacts')
    proof = json.loads((directory / 'proof/report.json').read_text())
    assert proof['complete'] and proof['baseline_restored'] and proof['snapshot_deleted']
    candidate = ROOT / 'artifacts/paid-common-raw-hybrid-v1'
    audit = json.loads((candidate / 'source-audit-readonly.json').read_text())
    assert audit['complete'] and audit['module'] == proof['candidate']
    original = ROOT / 'scripts/prove_paid_update_upgrade_guards.py'
    source = original.read_text()
    source = source.replace("R=Path(__file__).resolve().parents[1];", 'R=Path(' + repr(str(ROOT)) + ');')
    source = source.replace('6eydd-o3777-77775-aaama-cai', '4caro-hl777-77775-aaaba-cai')
    source = source.replace('artifacts/paid-update-v1/build-upgrade-diagnostic', 'artifacts/paid-common-raw-hybrid-v1/build')
    source = source.replace('artifacts/paid-update-v1/upgrade-guards-v1', str((directory / 'upgrade-guards').relative_to(ROOT)))
    frozen = directory / 'frozen-upgrade-guards.py'
    assert not frozen.exists() and not (directory / 'upgrade-guards').exists()
    compile(source, str(frozen), 'exec')
    frozen.write_text(source)
    files = [Path(__file__), original, frozen, candidate / 'source-audit-readonly.json', directory / 'proof/report.json']
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (directory / 'upgrade-guard-entry-hashes.json').write_text(json.dumps(manifest, indent=2) + '\n')
    exec(compile(source, str(frozen), 'exec'), dict(__file__=str(frozen), __name__='__main__'))


if __name__ == '__main__':
    main()
