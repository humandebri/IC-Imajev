#!/usr/bin/env python3
"""Compare exact prepacked Strassen and ordinary INT8 on the same local canister."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, decode, encode, is_instruction_limit
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--source', default='artifacts/kernel-default-v4-617')
    args = ap.parse_args()
    dest = ROOT / args.directory
    dest.mkdir(parents=True, exist_ok=True)
    pack_dir = ROOT / 'artifacts/strassen-prepacked'
    manifest = json.loads((pack_dir / 'manifest.json').read_text())
    source = ROOT / args.source
    report = json.loads((source / 'report.json').read_text())
    tensor = manifest['source_tensor']
    query = next(q for q in report['queries'] if q['op'] == 'lora_integer'
                 and decode((source / 'queries' / f"{q['index']:06d}.request.bin").read_bytes())[0]['tensor'] == tensor)
    source_path = source / 'queries' / f"{query['index']:06d}.request.bin"
    header, values = decode(source_path.read_bytes())
    values = values.reshape(header['dims'][0], header['dims'][2])
    module = ROOT / 'target/wasm32-unknown-unknown/release/imajev_inference.wasm'
    wasm_hash = hashlib.sha256(module.read_bytes()).hexdigest()
    t = Transport(manifest['model'], 'http://localhost:8001/', args.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, manifest['pack_hash'])
    cases = []
    try:
        verify_module(t, wasm_hash)
        for n in [1, 7, 8, 32, 64, 87]:
            outputs = {}
            case = {'tokens': n, 'dims': [n, 8192, 2560, 0]}
            for op in ['linear_integer_bf16', 'linear_strassen_bf16']:
                h = dict(header, op=op, pack_hash=manifest['pack_hash'],
                         dims=case['dims'], scalars=[], aux=[], encoding='bf16-exact')
                if op == 'linear_strassen_bf16':
                    h['aux'] = [tensor + '.strassen_i8', tensor + '.strassen_corrections']
                request = dest / f'{n}-{op}.request.bin'
                response = dest / f'{n}-{op}.response.bin'
                request.write_bytes(encode(h, values[:n]))
                try:
                    result = t.command({'op': 'step', 'input': str(request), 'output': str(response)})
                    _, output = decode(response.read_bytes())
                    outputs[op] = output
                    case[op] = result
                    if op == 'linear_strassen_bf16':
                        native = dest / f'{n}.native.bin'
                        subprocess.run([str(ROOT / 'target/release/primitive'), str(request), str(native),
                                        str(pack_dir / 'manifest.json'), str(pack_dir / 'pack.bin')], check=True)
                        _, expected = decode(native.read_bytes())
                        case['native_bitwise_equal'] = bool(np.array_equal(output.view(np.uint32), expected.view(np.uint32)))
                        assert case['native_bitwise_equal'], case
                except RuntimeError as e:
                    if not is_instruction_limit(e):
                        raise
                    case[op] = {'instruction_limit_exceeded': True, 'error': str(e)}
            if len(outputs) == 2:
                case['bitwise_equal'] = bool(np.array_equal(outputs['linear_integer_bf16'].view(np.uint32),
                                                            outputs['linear_strassen_bf16'].view(np.uint32)))
                assert case['bitwise_equal'], case
            cases.append(case)
            print(json.dumps(case), flush=True)
        verify_module(t, wasm_hash)
        (dest / 'report.json').write_text(json.dumps({
            'cases': cases, 'wasm_sha256': wasm_hash, 'canister': args.canister,
            'pack_hash': manifest['pack_hash'],
            'source_request_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'scope': 'Pure base Q projection only; not full model accuracy or query count',
        }, indent=2) + '\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
