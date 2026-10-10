#!/usr/bin/env python3
"""Build the 5-token browser runtime from the pinned production kernel lineage.

Never deploys. Retains the validated projection bodies, imports the current
paid API, and updates the prefix and streaming suffix bounds (maximum 91).
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from build_paid_message_checkpoint import projection_bodies, sha
from build_paid_contract import copy_paid_sources, align_upgrade_metadata
from canister_api_names import rename_directory_api_methods
from build_prefix_contract import align_prefix_contract
from generate_wasm_candid import generate_candid

ROOT = Path(__file__).resolve().parents[1]
STREAM_FILES = ('mlp_stream.rs', 'delta_mlp_start.rs', 'attention_mlp_stream.rs',
                'mlp_attention_finish.rs', 'mlp_pipeline.rs', 'mlp_delta_stream.rs',
                'delta_head_continue.rs', 'delta_full_log.rs', 'attention_full.rs', 'prefix_start.rs')


def install_paid_sources(directory, canonical):
    """Keep the paid ABI current without replacing validated numerical kernels."""
    copy_paid_sources(directory, canonical)
    align_upgrade_metadata(directory)
    scheduler = directory / 'update_inference.rs'
    text = scheduler.read_text()
    old = 'const STOP: u64 = 34_000_000_000;'
    if text.count(old) != 1:
        raise ValueError('missing frozen worker budget')
    text = text.replace(old, 'pub(super) const WORKER_BUDGET: u64 = 30_000_000_000;', 1)
    text = text.replace('<STOP', '<WORKER_BUDGET')
    text += '\npub(super) fn progress_limit(_: usize) -> u64 { 64 }\n'
    scheduler.write_text(text)
    return align_prefix_contract(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source_root.resolve()
    parent = source / 'artifacts/common-prefix27/latest-public-query-build-v2'
    runtime_parent = source / 'artifacts/update-rank7-prepare8-unrolled-v1/build'
    base = json.loads((parent / 'report.json').read_text())
    runtime = json.loads((runtime_parent / 'report.json').read_text())
    assert sha(parent / 'full.wasm') == base['wasm_sha256']
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copytree(runtime_parent / 'runtime', directory / 'runtime')
    for name in STREAM_FILES:
        path = directory / 'runtime' / name
        text = path.read_text()
        updated = text.replace('1..=89', '1..=91').replace('n > 89', 'n > 91').replace('n>89', 'n>91').replace('*n<=89', '*n<=91').replace('n>90', 'n>91')
        if name == 'mlp_attention_finish.rs':
            updated = updated.replace('!(1..=91).contains(&n)', '!(1..=if stream(r){91}else{89}).contains(&n)')
        assert updated != text, name
        path.write_text(updated)
    for path in parent.glob('*.rs'):
        shutil.copyfile(path, directory / path.name)
    for name in ('query_mlp_delta.rs', 'query_attention_mlp_front.rs'):
        path = directory / name
        text = path.read_text()
        assert text.count('1..=69') == 1, name
        path.write_text(text.replace('1..=69', '1..=91'))
    lib = directory / 'lib.rs'
    text = lib.read_text()
    old = '|| r.dims.len() != 3 || !(1..=89).contains(&r.dims[0])'
    assert text.count(old) == 1
    lib.write_text(text.replace(old, '|| r.dims.len() != 3 || !(1..=91).contains(&r.dims[0])'))
    prefix_tokens = install_paid_sources(directory, ROOT / 'canisters/inference/src')
    if prefix_tokens != 5:
        raise ValueError('browser release requires a five-token prefix')
    rename_directory_api_methods(directory)
    lib = directory / "lib.rs"
    if "use query_attention_mlp_front::AttentionMlpFrontMeasurement;" not in lib.read_text():
        lib.write_text(lib.read_text() + "\nuse query_attention_mlp_front::AttentionMlpFrontMeasurement;\n")
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(directory),
               CARGO_PKG_NAME='imajev-inference', CARGO_PKG_VERSION='0.1.0',
               CARGO_PKG_VERSION_MAJOR='0', CARGO_PKG_VERSION_MINOR='1',
               CARGO_PKG_VERSION_PATCH='0', CARGO_PKG_VERSION_PRE='',
               CARGO_CRATE_NAME='imajev_inference')
    runtime_command = runtime['runtime_command'][:]
    runtime_command[runtime_command.index('--edition=2021') + 1] = str(directory / 'runtime/lib.rs')
    runtime_command[runtime_command.index('-o') + 1] = str(directory / 'libimajev_runtime.rlib')
    command = base['command'][:]
    command[command.index('--edition=2021') + 1] = str(directory / 'lib.rs')
    command[command.index('-o') + 1] = str(directory / 'raw.wasm')
    for index, value in enumerate(command):
        if value.startswith('imajev_runtime='):
            command[index] = 'imajev_runtime=' + str(directory / 'libimajev_runtime.rlib')
    inputs = list(directory.glob('*.rs')) + list((directory / 'runtime').glob('*.rs'))
    inputs += [ROOT / 'scripts' / name for name in
               ('build_paid_contract.py', 'build_prefix_contract.py', 'canister_api_names.py',
                'generate_wasm_candid.py', 'check_canister_api_exports.py')]
    hashes = {str(path): sha(path) for path in inputs}
    with (directory / 'compiler.log').open('w') as log:
        for cmd in (runtime_command, command):
            subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=log, stderr=log)
    previous = directory / 'raw.wasm'
    patches = []
    donors = projection_bodies((parent / 'full.wasm').read_bytes(), base['patches'])
    for index, (entry, donor) in enumerate(zip(base['patches'], donors, strict=True)):
        donor_path = directory / f'donor-{index}.wasm'
        donor_path.write_bytes(donor)
        output = directory / ('full.wasm' if index == len(base['patches']) - 1 else f'patched-{index}.wasm')
        row = json.loads(subprocess.check_output([
            str(source / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),
            str(previous), str(donor_path), str(output), entry['export']], text=True))
        assert row['wasmparser_validation'] and row['replacement_body_sha256'] == entry['replacement_body_sha256']
        patches.append(row)
        previous = output
    assert len(patches) == 34
    assert hashes == {name: sha(Path(name)) for name in hashes}
    api = generate_candid(directory / 'full.wasm', directory / 'service.did')
    report = dict(api=api, wasm_sha256=sha(directory / 'full.wasm'), parent=base['wasm_sha256'],
                  prefix_tokens=5, browser_max_suffix=91, command=command,
                  runtime_command=runtime_command, source_hashes=hashes, patches=patches,
                  build_script_sha256=sha(Path(__file__)),
                  all34_projection_bodies_equal_to_validated_parent=True,
                  full_inference_verified=False, deployed=False)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for path in list(directory.glob('patched-*.wasm')) + list(directory.glob('donor-*.wasm')) + [directory / 'raw.wasm']:
        path.unlink()
    print(json.dumps(dict(module=report['wasm_sha256'], prefix=5, max_suffix=91)), flush=True)


if __name__ == '__main__':
    main()
