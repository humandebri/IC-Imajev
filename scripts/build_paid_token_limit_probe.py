#!/usr/bin/env python3
"""Build a local-only input-limit probe without changing production sources."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from build_paid_message_checkpoint import projection_bodies, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    parent = a.source_root / 'artifacts/common-prefix27/latest-public-query-build-v2'
    base = json.loads((parent / 'report.json').read_text())
    assert sha(parent / 'full.wasm') == base['wasm_sha256']
    for name, digest in base['dependency_hashes'].items():
        assert sha(a.source_root / name) == digest, name
    d = a.directory.resolve()
    d.mkdir(parents=True, exist_ok=False)
    for source in parent.glob('*.rs'):
        shutil.copyfile(source, d / source.name)
    # Retain the reviewed counter-zero checkpoint from the working source.
    shutil.copyfile(ROOT / 'canisters/inference/src/paid_inference.rs', d / 'paid_inference.rs')
    modifications = {
        'paid_inference.rs': [
            ('const MAX_STEPS:u32=8;', 'const MAX_STEPS:u32=64;'),
            ('const INPUT_LIMIT:usize=116;', 'const INPUT_LIMIT:usize=500;'),
            ('max_steps:if n>229 {64}else if n>89 {32}else{MAX_STEPS}', 'max_steps:MAX_STEPS'),
        ],
        'update_inference.rs': [
            ('const STOP: u64 = 34_000_000_000;', 'pub(super) const WORKER_BUDGET: u64 = 30_000_000_000;'),
            ('<STOP', '<WORKER_BUDGET'),
            ('(1..=89).contains(&ids.len())', '(1..=473).contains(&ids.len())'),
        ],
    }
    for name, replacements in modifications.items():
        text = (d / name).read_text()
        for old, new in replacements:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        if name == 'update_inference.rs':
            text += '\npub(super) fn progress_limit(_:usize)->u64 {64}\n'
        (d / name).write_text(text)
    command = base['command'][:]
    command[command.index('--edition=2021') + 1] = str(d / 'lib.rs')
    command[command.index('-o') + 1] = str(d / 'raw.wasm')
    command += ['--cfg', 'feature="paid-update-diagnostics"']
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(d), CARGO_PKG_NAME='imajev-inference',
               CARGO_PKG_VERSION='0.1.0', CARGO_PKG_VERSION_MAJOR='0',
               CARGO_PKG_VERSION_MINOR='1', CARGO_PKG_VERSION_PATCH='0',
               CARGO_PKG_VERSION_PRE='', CARGO_CRATE_NAME='imajev_inference')
    with (d / 'compiler.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=log)
    previous = d / 'raw.wasm'
    patches = []
    for i, (entry, donor) in enumerate(zip(base['patches'], projection_bodies(
            (parent / 'full.wasm').read_bytes(), base['patches']), strict=True)):
        source = d / f'donor-{i}.wasm'
        source.write_bytes(donor)
        target = d / ('full.wasm' if i == len(base['patches']) - 1 else f'patched-{i}.wasm')
        row = json.loads(subprocess.check_output([
            str(a.source_root / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),
            str(previous), str(source), str(target), entry['export']], text=True))
        assert row['wasmparser_validation']
        assert row['replacement_body_sha256'] == entry['replacement_body_sha256']
        patches.append(row)
        previous = target
    report = dict(module=sha(d / 'full.wasm'), parent=base['wasm_sha256'],
                  modifications=modifications, command=command, patches=patches,
                  dependencies=base['dependency_hashes'],
                  sources={str(f): sha(f) for f in d.glob('*.rs')},
                  local_only=True, numerical_runtime_unchanged=True,
                  all34_projection_bodies_equal=True)
    (d / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(module=report['module'], patches=len(patches))), flush=True)


if __name__ == '__main__':
    main()
