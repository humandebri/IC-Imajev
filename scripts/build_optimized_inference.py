#!/usr/bin/env python3
"""Build the source-only optimized candidate. Never deploys or approves migration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from generate_wasm_candid import generate_candid

ROOT = Path(__file__).resolve().parents[1]
RELEASE_PROFILE = {'opt-level': '3', 'lto': '"thin"', 'codegen-units': '1',
                   'panic': '"abort"', 'overflow-checks': 'true', 'debug': 'false',
                   'debug-assertions': 'false', 'strip': '"none"', 'incremental': 'false'}


def build_environment(source, directory, inherited):
    env = dict(inherited)
    for key in list(env):
        if key in {'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS',
                   'RUSTC', 'RUSTDOC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER',
                   'CARGO_BUILD_TARGET'} or key.startswith('CARGO_PROFILE_'):
            env.pop(key)
    cargo_home = Path(env.get('CARGO_HOME', Path.home() / '.cargo')).resolve()
    flags = [f'--remap-path-prefix={source}=/source', f'--remap-path-prefix={cargo_home}=/cargo', '-C', 'target-feature=+simd128']
    env.update(RUSTUP_TOOLCHAIN='1.97.1', CARGO_TARGET_DIR=str(directory / 'cargo'),
               CARGO_INCREMENTAL='0', CARGO_ENCODED_RUSTFLAGS='\x1f'.join(flags))
    return env, flags


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path, help='Fresh output directory')
    parser.add_argument('--offline', action='store_true', help='Use previously downloaded Cargo dependencies')
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument('--kernel-contract-tests', action='store_true', help='Verification-only Wasm with synthetic arithmetic probes; never a migration candidate')
    verification.add_argument('--paid-contract-tests', action='store_true', help='Verification-only Wasm with owner fault/upgrade probes; never a migration candidate')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if directory.exists():
        raise SystemExit('Refusing to reuse an output directory')
    env, flags = build_environment(ROOT, directory, os.environ)
    rustc = subprocess.check_output(['rustup', 'which', '--toolchain', '1.97.1', 'rustc'], env=env, text=True).strip()
    cargo = subprocess.check_output(['rustup', 'which', '--toolchain', '1.97.1', 'cargo'], env=env, text=True).strip()
    env['RUSTC'] = rustc
    version = subprocess.check_output([rustc, '--version'], env=env, text=True).strip()
    compiler_details = subprocess.check_output([rustc, '-vV'], env=env, text=True)
    host = next(line.removeprefix('host: ') for line in compiler_details.splitlines() if line.startswith('host: '))
    if version.split()[1] != '1.97.1':
        raise SystemExit('rustc 1.97.1 is required')
    # Do not allow inherited flags/features to change the production configuration.
    files = [ROOT / name for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'scripts/build_optimized_inference.py', 'scripts/generate_wasm_candid.py', 'scripts/check_canister_api_exports.py', 'scripts/canister_api_names.py', 'scripts/prefix_state_cache.rs', 'frontend/scripts/check-optimized-kernels.mjs']]
    for name in ['crates', 'canisters', 'tools/wasm-kernel-patch', 'kernels/optimized']:
        files.extend(p for p in (ROOT / name).rglob('*') if p.is_file() and 'target' not in p.parts and (p.suffix in {'.rs', '.toml', '.json'} or p.name.endswith('.wat.gz')))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}
    directory.mkdir(parents=True)
    common = [cargo, 'build', '--locked'] + (['--offline'] if args.offline else [])
    for key, value in RELEASE_PROFILE.items():
        common.extend(['--config', f'profile.release.{key}={value}'])
    features = ['kernel-contract-tests'] if args.kernel_contract_tests else ['optimized-release']
    if args.paid_contract_tests:
        features.append('paid-update-diagnostics')
    with (directory / 'compiler.log').open('w') as log:
        subprocess.run(common + ['--release', '--target', 'wasm32-unknown-unknown', '-p', 'imajev-inference', '--no-default-features', '--features', ','.join(features)], cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
        # Build the patcher from the same public source; never use a retained binary.
        native = dict(env); native.pop('CARGO_ENCODED_RUSTFLAGS')
        native['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join(flags[:2])
        subprocess.run(common + ['--release', '--target', host, '-p', 'imajev-wasm-kernel-patch'], cwd=ROOT, env=native, stdout=log, stderr=log, check=True)
    raw = directory / 'raw.wasm'
    shutil.copyfile(directory / 'cargo/wasm32-unknown-unknown/release/imajev_inference.wasm', raw)
    executable = 'imajev-wasm-kernel-patch.exe' if 'windows' in host else 'imajev-wasm-kernel-patch'
    result = json.loads(subprocess.check_output([str(directory / 'cargo' / host / 'release' / executable), str(raw), str(ROOT / 'kernels/optimized'), str(directory / 'optimized.wasm')], text=True))
    api = generate_candid(directory / 'optimized.wasm', directory / 'service.did', diagnostics=args.paid_contract_tests, embed_public=True)
    if hashes != {name: digest(ROOT / name) for name in hashes}:
        raise RuntimeError('Source changed during build')
    kernel_tests = None
    if args.kernel_contract_tests:
        kernel_tests = json.loads(subprocess.check_output(['node', str(ROOT / 'frontend/scripts/check-optimized-kernels.mjs'), str(directory / 'optimized.wasm')], text=True))
    report = dict(kernel_tests=kernel_tests, format=1, build_complete=True, release_eligible=False,
                  reason='Numerical, local IC budget and performance gates must pass before migration.',
                  rustc=version, release_profile=RELEASE_PROFILE, target='wasm32-unknown-unknown', features=features, verification_only=args.kernel_contract_tests or args.paid_contract_tests,
                  flags=['--remap-path-prefix=<source>=/source', '--remap-path-prefix=<cargo-home>=/cargo', '-C target-feature=+simd128'],
                  raw_sha256=digest(raw), module=digest(directory / 'optimized.wasm'),
                  source_hashes=hashes, patches=result['patches'], api=api,
                  full_inference_verified=False, source_only_build=True)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['module', 'build_complete', 'release_eligible']}, indent=2))


if __name__ == '__main__':
    main()
