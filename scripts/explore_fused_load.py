#!/usr/bin/env python3
"""Compare an exact SIMD load candidate on the explicitly selected local canister.

Temporarily builds/upgrades the selected experimental canister. Restores source
and baseline Wasm in finally; rejects dirty baseline source or a module mismatch.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--profile', action='store_true', help='Measure current kernel spans instead of changing its load')
    args = ap.parse_args()
    source = ROOT / 'crates/imajev-runtime/src/int8_kernel.rs'
    wasm = ROOT / 'target/wasm32-unknown-unknown/release/imajev_inference.wasm'
    dest = ROOT / 'artifacts/exploration' / ('current-profile' if args.profile else 'fused-load')
    dest.mkdir(parents=True, exist_ok=True)
    original = source.read_bytes()
    original_wasm = wasm.read_bytes()
    if original != subprocess.check_output(['git', 'show', 'HEAD:crates/imajev-runtime/src/int8_kernel.rs'], cwd=ROOT):
        raise ValueError('Kernel differs from committed baseline')
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    baseline = dest / 'baseline.wasm'
    baseline.write_bytes(original_wasm)
    t = Transport(manifest['model'], 'http://localhost:8001/', args.canister, str(ROOT / 'artifacts/imajev-local.pem'), dest / 'verify', manifest['pack_hash'])
    try:
        verify_module(t, hashlib.sha256(original_wasm).hexdigest())
    finally:
        t.close()
    def install(path):
        subprocess.run(['icp', 'canister', 'install', args.canister, '--mode', 'upgrade', '--wasm', str(path), '--network', 'local', '--identity', 'imajev-local'], cwd=ROOT, check=True)
    def measure(name, profile=False):
        command = [sys.executable, str(ROOT / 'scripts/profile_bottlenecks.py'), '--phase', name, '--source-directory', 'artifacts/layout-full-617', '--integer-only', '--canister', args.canister]
        if not profile:
            command.append('--normal-only')
        subprocess.run(command, cwd=ROOT, check=True)
    try:
        install(baseline)
        measure('explore-load-baseline')
        before = 'i16x8_extend_low_i8x16(v128_load64_zero(wp[j].add(c + g * 8).cast()))'
        after = 'i16x8_load_extend_i8x8(wp[j].add(c + g * 8).cast())'
        text = original.decode()
        if text.count(before) != 1:
            raise ValueError('Unexpected baseline load expression')
        source.write_text(text if args.profile else text.replace(before, after))
        command = ['cargo', 'build', '--release', '--target', 'wasm32-unknown-unknown', '-p', 'imajev-inference', '--offline']
        if args.profile:
            command.extend(['--features', 'instruction-profile'])
        subprocess.run(command, cwd=ROOT, check=True)
        candidate = dest / 'candidate.wasm'
        candidate.write_bytes(wasm.read_bytes())
        (dest / 'candidate.rs').write_bytes(source.read_bytes())
        install(candidate)
        measure('explore-current-profile' if args.profile else 'explore-load-combined', profile=args.profile)
    finally:
        source.write_bytes(original)
        wasm.write_bytes(original_wasm)
        install(baseline)
        t = Transport(manifest['model'], 'http://localhost:8001/', args.canister, str(ROOT / 'artifacts/imajev-local.pem'), dest / 'restored', manifest['pack_hash'])
        try:
            verify_module(t, hashlib.sha256(original_wasm).hexdigest())
        finally:
            t.close()
        print('Restored baseline source and deployed module.', flush=True)


if __name__ == '__main__':
    main()
