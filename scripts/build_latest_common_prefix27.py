#!/usr/bin/env python3
"""Build public prefix27 inference from the latest validated optimized runtime."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from build_paid_message_checkpoint import projection_bodies, sha
from public_query_build import expose_public_queries

ROOT = Path(__file__).resolve().parents[1]


def prefix27_scheduler(text):
    # Keep prepared Delta states and copy/hash optimizations from the parent.
    for old, new in [
        ('if !(1..=132).contains(&p)', 'if p!=27'),
        ('if g.banks.len()>=2', 'if !g.banks.is_empty()'),
    ]:
        if text.count(old) != 1:
            raise ValueError(f'unexpected scheduler: {old}')
        text = text.replace(old, new, 1)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    parent = ROOT / 'artifacts/paid-stack-carry-projection-v1/build'
    pointer_path = parent.parent / 'validated-proof-pointer.json'
    pointer = json.loads(pointer_path.read_text())
    base = json.loads((parent / 'report.json').read_text())
    assert pointer['complete'] and pointer['validated_best']
    assert sha(parent / 'full.wasm') == base['wasm_sha256'] == pointer['module']
    for group in ['source_hashes', 'dependency_hashes']:
        for name, digest in base[group].items():
            assert sha(ROOT / name) == digest, name
    for name, digest in pointer['hashes'].items():
        assert sha(ROOT / name) == digest, name
    directory = (ROOT / args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for path in parent.glob('*.rs'):
        shutil.copyfile(path, directory / path.name)
    for name in ['paid_inference.rs', 'paid_types.rs']:
        shutil.copyfile(ROOT / 'canisters/inference/src' / name, directory / name)
    scheduler = directory / 'update_inference.rs'
    scheduler.write_text(prefix27_scheduler(scheduler.read_text()))
    expose_public_queries(directory)
    command = base['command'][:]
    # Produce the normal module, without owner diagnostic/fault endpoints.
    i = command.index('feature="paid-update-diagnostics"')
    assert command[i - 1] == '--cfg'
    del command[i - 1:i + 1]
    command[command.index('--edition=2021') + 1] = str(directory / 'lib.rs')
    command[command.index('-o') + 1] = str(directory / 'raw.wasm')
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(directory),
               CARGO_PKG_NAME='imajev-inference', CARGO_PKG_VERSION='0.1.0',
               CARGO_PKG_VERSION_MAJOR='0', CARGO_PKG_VERSION_MINOR='1',
               CARGO_PKG_VERSION_PATCH='0', CARGO_PKG_VERSION_PRE='',
               CARGO_CRATE_NAME='imajev_inference')
    files = [Path(__file__), ROOT / 'scripts/build_common_prefix27.py', ROOT / 'scripts/build_paid_message_checkpoint.py',
             ROOT / 'scripts/public_query_build.py', pointer_path,
             parent / 'report.json', parent / 'full.wasm'] + list(directory.glob('*.rs'))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    with (directory / 'compiler.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=log)
    previous = directory / 'raw.wasm'
    patches = []
    donors = projection_bodies((parent / 'full.wasm').read_bytes(), base['patches'])
    for i, (entry, donor) in enumerate(zip(base['patches'], donors, strict=True)):
        source = directory / f'projection-donor-{i}.wasm'
        source.write_bytes(donor)
        target = directory / ('full.wasm' if i == len(base['patches']) - 1 else f'patched{i}.wasm')
        row = json.loads(subprocess.check_output([
            str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),
            str(previous), str(source), str(target), entry['export']], text=True))
        assert row['wasmparser_validation']
        assert row['replacement_body_sha256'] == entry['replacement_body_sha256']
        patches.append(row)
        previous = target
    assert len(patches) == 34
    assert hashes == {name: sha(ROOT / name) for name in hashes}
    report = dict(wasm_sha256=sha(directory / 'full.wasm'), parent=base['wasm_sha256'],
                  command=command, patches=patches, source_hashes=hashes,
                  dependency_hashes=base['dependency_hashes'],
                  public_queries=True, prefix_tokens=27,
                  all34_projection_bodies_equal_to_validated_parent=True,
                  scope='Build only. Optimized parent retained; combined prefix27 inference and instruction counts require fresh verification. No deployment.',
                  full_inference_verified=False)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(module=report['wasm_sha256'], patches=len(patches))), flush=True)


if __name__ == '__main__':
    main()
