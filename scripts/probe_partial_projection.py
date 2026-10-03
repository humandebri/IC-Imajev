#!/usr/bin/env python3
"""Measure real Q projection requests on a sealed partial diagnostic pack."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, decode
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
    packdir = ROOT / 'artifacts/strassen-prepacked'
    m = json.loads((packdir / 'manifest.json').read_text())
    module_hash = hashlib.sha256((ROOT / args.wasm).read_bytes()).hexdigest()
    source = packdir / 'probe'
    before = json.loads((ROOT / args.before / 'report.json').read_text()) if args.before else None
    t = Transport(m['model'], 'http://localhost:8001/', args.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash'])
    cases = []
    try:
        verify_module(t, module_hash)
        for index, n in enumerate([1, 7, 8, 32, 64, 87]):
            request = source / f'{n}-linear_integer_bf16.request.bin'
            header, _ = decode(request.read_bytes())
            assert header['pack_hash'] == m['pack_hash']
            assert header['op'] == 'linear_integer_bf16'
            assert header['tensor'] == m['source_tensor']
            assert header['dims'] == [n, 8192, 2560, 0]
            response = dest / f'{n}.response.bin'
            result = t.command(dict(op='step', input=str(request), output=str(response)))
            _, got = decode(response.read_bytes())
            _, expected = decode((source / f'{n}-linear_integer_bf16.response.bin').read_bytes())
            assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))
            native = dest / f'{n}.native.bin'
            subprocess.run([str(ROOT / 'target/release/primitive'), str(request), str(native),
                            str(packdir / 'manifest.json'), str(packdir / 'pack.bin')], check=True)
            _, scalar = decode(native.read_bytes())
            assert np.array_equal(got.view(np.uint32), scalar.view(np.uint32))
            case = dict(tokens=n, bitwise_equal=True, native_bitwise_equal=True,
                        request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(), **result)
            if before:
                old = before['cases'][index]
                assert old['request_sha256'] == case['request_sha256']
                case['before_instructions'] = old['ok']['instructions']
                case['instruction_reduction'] = 1-result['ok']['instructions']/old['ok']['instructions']
            cases.append(case)
            print(json.dumps(case), flush=True)
        verify_module(t, module_hash)
        (dest / 'report.json').write_text(json.dumps(dict(cases=cases, wasm_sha256=module_hash,
                    pack_hash=m['pack_hash'], scope='Pure Q projection; full graph validation required'), indent=2)+'\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
