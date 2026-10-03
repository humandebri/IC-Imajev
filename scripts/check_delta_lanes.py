#!/usr/bin/env python3
"""Measure Delta SIMD grouping, with exact outputs and scalar native oracle."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, encode, decode
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--before')
    args = ap.parse_args()
    dest = ROOT / args.directory
    dest.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'artifacts/strassen-prepacked/manifest.json'
    pack = manifest.with_name('pack.bin')
    m = json.loads(manifest.read_text())
    sha = hashlib.sha256((ROOT / args.wasm).read_bytes()).hexdigest()
    rng = np.random.default_rng(671)
    t = Transport(m['model'], 'http://localhost:8001/', args.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash'])
    cases = []
    try:
        verify_module(t, sha)
        for n in [1, 45, 87]:
            for dv in [4, 12, 16, 128]:
                dk = 128
                q = rng.normal(0, .08, n * dk).astype(np.float32)
                k = rng.normal(0, .08, n * dk).astype(np.float32)
                v = rng.normal(0, .2, n * dv).astype(np.float32)
                g = rng.uniform(.7, 1, n).astype(np.float32)
                beta = rng.uniform(.01, .99, n).astype(np.float32)
                state = rng.normal(0, .1, dk * dv).astype(np.float32)
                x = np.concatenate([q, k, v, g, beta, state])
                h = dict(version=1, model=m['model'], pack_hash=m['pack_hash'], input_hash='0'*64,
                         step=0, op='delta_bf16', tensor='', dims=[n,dk,dv], scalars=[], encoding='bf16-exact')
                name = f'{n}-{dv}'
                request = dest / f'{name}.request.bin'
                response = dest / f'{name}.response.bin'
                request.write_bytes(encode(h, x))
                result = t.command(dict(op='step', input=str(request), output=str(response)))
                native = dest / f'{name}.native.bin'
                subprocess.run([str(ROOT / 'target/release/primitive'), str(request), str(native),
                                str(manifest), str(pack)], check=True)
                _, got = decode(response.read_bytes())
                _, expected = decode(native.read_bytes())
                assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), name
                case = dict(dims=h['dims'], native_bitwise_equal=True, **result)
                if args.before:
                    before = ROOT / args.before
                    assert request.read_bytes() == (before / request.name).read_bytes()
                    _, old = decode((before / response.name).read_bytes())
                    case['before_bitwise_equal'] = bool(np.array_equal(got.view(np.uint32), old.view(np.uint32)))
                    assert case['before_bitwise_equal'], name
                cases.append(case)
                print(json.dumps(case), flush=True)
        verify_module(t, sha)
        (dest / 'report.json').write_text(json.dumps(dict(cases=cases, wasm_sha256=sha,
                                                        scope='Synthetic recurrence only; full graph validation required'), indent=2)+'\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
