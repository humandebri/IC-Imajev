#!/usr/bin/env python3
"""Build public prefix27 inference from the latest validated optimized runtime."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from build_prefix_contract import align_prefix_contract
from build_paid_contract import copy_paid_sources, align_upgrade_metadata
from generate_wasm_candid import generate_candid

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
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Checkout containing the frozen runtime and dependency artifacts')
    parser.add_argument('--legacy-116', action='store_true',
                        help='Build without token chunks (legacy flag name; current prefix + 89-token suffix)')
    args = parser.parse_args()
    artifact_root = args.source_root.resolve()
    if not args.legacy_116:
        subprocess.run([sys.executable, str(ROOT / 'scripts/build_paid_token_chunks.py'),
                        '--source-root', str(artifact_root), '--directory', str((ROOT / args.directory).resolve())], check=True)
        return
    parent = artifact_root / 'artifacts/paid-stack-carry-projection-v1/build'
    pointer_path = parent.parent / 'validated-proof-pointer.json'
    pointer = json.loads(pointer_path.read_text())
    base = json.loads((parent / 'report.json').read_text())
    assert pointer['complete'] and pointer['validated_best']
    assert sha(parent / 'full.wasm') == base['wasm_sha256'] == pointer['module']
    for group in ['source_hashes', 'dependency_hashes']:
        for name, digest in base[group].items():
            assert sha(artifact_root / name) == digest, name
    for name, digest in pointer['hashes'].items():
        assert sha(artifact_root / name) == digest, name
    directory = (ROOT / args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for path in parent.glob('*.rs'):
        shutil.copyfile(path, directory / path.name)
    copy_paid_sources(directory, ROOT / 'canisters/inference/src')
    align_upgrade_metadata(directory)
    scheduler = directory / 'update_inference.rs'
    text = prefix27_scheduler(scheduler.read_text())
    old = 'const STOP: u64 = 34_000_000_000;'
    assert text.count(old) == 1
    text = text.replace(old, 'pub(super) const WORKER_BUDGET: u64 = 30_000_000_000;').replace('<STOP', '<WORKER_BUDGET')
    text += "\npub(super) fn progress_limit(_: usize) -> u64 { 64 }\n"
    scheduler.write_text(text)
    prefix_tokens = align_prefix_contract(directory)
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
             ROOT / 'scripts/build_paid_contract.py', ROOT / 'scripts/generate_wasm_candid.py', ROOT / 'scripts/check_canister_api_exports.py', ROOT / 'scripts/public_query_build.py', ROOT / 'scripts/build_prefix_contract.py', ROOT / 'scripts/canister_api_names.py', pointer_path,
             parent / 'report.json', parent / 'full.wasm'] + list(directory.glob('*.rs'))
    hashes = {str(p): sha(p) for p in files}
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
            str(artifact_root / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),
            str(previous), str(source), str(target), entry['export']], text=True))
        assert row['wasmparser_validation']
        assert row['replacement_body_sha256'] == entry['replacement_body_sha256']
        patches.append(row)
        previous = target
    assert len(patches) == 34
    assert hashes == {name: sha(ROOT / name) for name in hashes}
    api = generate_candid(directory / 'full.wasm', directory / 'service.did')
    report = dict(api=api, wasm_sha256=sha(directory / 'full.wasm'), parent=base['wasm_sha256'],
                  command=command, patches=patches, source_hashes=hashes,
                  dependency_hashes=base['dependency_hashes'],
                  public_queries=True, prefix_tokens=prefix_tokens,
                  all34_projection_bodies_equal_to_validated_parent=True,
                  scope='Build only. Optimized parent retained; combined prefix27 inference and instruction counts require fresh verification. No deployment.',
                  full_inference_verified=False)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(module=report['wasm_sha256'], patches=len(patches))), flush=True)


if __name__ == '__main__':
    main()
