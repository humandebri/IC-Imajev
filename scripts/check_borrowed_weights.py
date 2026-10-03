#!/usr/bin/env python3
"""Compare ordinary queries before/after immutable weight warmup, same module."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, decode, encode
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--extended', action='store_true')
    ap.add_argument('--baseline', help='Prior same-input preparation probe directory')
    ap.add_argument('--native', default='artifacts/strassen-wide/default-primitive')
    args = ap.parse_args()
    dest = ROOT / args.directory
    dest.mkdir(parents=True, exist_ok=True)
    packdir = ROOT / 'artifacts/strassen-prepacked'
    m = json.loads((packdir / 'manifest.json').read_text())
    sha = hashlib.sha256((ROOT / args.wasm).read_bytes()).hexdigest()
    t = Transport(m['model'], 'http://localhost:8001/', args.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash'])
    source_tensor = next(x for x in m['tensors'] if x['name'] == m['source_tensor'])
    fixtures = []
    for n, rows, start in [(n, 8192, 0) for n in [1, 7, 8, 32, 64, 87]] + [(1, 16, 8), (7, 24, 4080), (87, 16, 8176)]:
        source = packdir / 'probe' / f'{n}-linear_integer_bf16'
        header, values = decode(source.with_suffix('.request.bin').read_bytes())
        assert header['pack_hash'] == m['pack_hash'] and header['tensor'] == m['source_tensor']
        assert header['dims'] == [n, 8192, 2560, 0]
        header['dims'] = [n, rows, 2560, start]
        request = dest / f'{n}-{rows}-{start}.request.bin'
        request.write_bytes(encode(header, values))
        expected = decode(source.with_suffix('.response.bin').read_bytes())[1].reshape(n, 8192)[:, start:start+rows].copy().ravel()
        native = dest / f'{n}-{rows}-{start}.native.bin'
        subprocess.run([str(ROOT / args.native), str(request), str(native),
                        str(packdir / 'manifest.json'), str(packdir / 'pack.bin')], check=True)
        scalar = decode(native.read_bytes())[1]
        assert np.array_equal(expected.view(np.uint32), scalar.view(np.uint32))
        fixtures.append((header, request, expected))
    if args.extended:
        for label, n in [('prefix', 45), ('insufficient', 80), ('maximum', 89), ('normal', 132)]:
            graph = ROOT / f'artifacts/quantized-invariants-v2-{label}'
            report = json.loads((graph / 'report.json').read_text())
            query = next(q for q in report['queries'] if q['op'] == 'lora_integer' and q['tensor'] == m['source_tensor'])
            header, values = decode((graph / 'queries' / f"{query['index']:06d}.request.bin").read_bytes())
            assert header['model'] == m['model'] and header['pack_hash'] == m['source_pack_hash']
            assert header['dims'][0] == n and header['dims'][2] == 2560
            rows = min(8192, 900000 // n // 8 * 8)
            header.update(pack_hash=m['pack_hash'], op='linear_integer_bf16', dims=[n, rows, 2560, 0], aux=[], scalars=[])
            request = dest / f'{n}-{rows}-0.request.bin'
            request.write_bytes(encode(header, values))
            native = request.with_suffix('.native.bin')
            subprocess.run([str(ROOT / args.native), str(request), str(native),
                            str(packdir / 'manifest.json'), str(packdir / 'pack.bin')], check=True)
            expected = decode(native.read_bytes())[1]
            fixtures.append((header, request, expected))
    prior = json.loads((ROOT / args.baseline / 'report.json').read_text()) if args.baseline else None
    results = {}
    def probe(phase):
        cases = []
        for header, request, expected in fixtures:
            dims = header['dims']
            response = dest / f"{phase}-{'-'.join(map(str, dims))}.response.bin"
            result = t.command(dict(op='step', input=str(request), output=str(response)))
            got = decode(response.read_bytes())[1]
            assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), (phase, dims)
            if phase == 'warm':
                assert result['ok']['stable_read_bytes'] == 0, (phase, dims)
            else:
                assert result['ok']['stable_read_bytes'] == dims[1] * (dims[2] + 4), (phase, dims)
            case = dict(dims=dims, bitwise_equal=True, native_bitwise_equal=True,
                        request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(), **result)
            if phase != 'cold':
                old = results['cold'][len(cases)]['ok']['instructions']
                case['instruction_reduction'] = 1-result['ok']['instructions']/old
            if prior:
                before = prior['phases'][phase][len(cases)]
                assert before['request_sha256'] == case['request_sha256']
                case['prior_module_instructions'] = before['ok']['instructions']
                case['kernel_instruction_reduction'] = 1-result['ok']['instructions']/before['ok']['instructions']
            cases.append(case)
            print(json.dumps(dict(phase=phase, **case)), flush=True)
        results[phase] = cases
    try:
        verify_module(t, sha)
        initial = t.command(dict(op='weight_cache_status'))
        assert initial['ok']['cache']['bytes'] == 0
        probe('cold')
        warm = t.command(dict(op='warm_weights', name=m['source_tensor']))
        assert warm['ok']['cache']['bytes'] == source_tensor['bytes']
        duplicate = t.command(dict(op='warm_weights', name=m['source_tensor']))
        assert duplicate['ok']['cache']['bytes'] == source_tensor['bytes']
        probe('warm')
        clear = t.command(dict(op='clear_weight_cache'))
        assert clear['ok']['cache']['bytes'] == 0
        probe('cleared')
        final = t.command(dict(op='weight_cache_status'))
        assert final['ok']['cache']['bytes'] == 0
        verify_module(t, sha)
        (dest / 'report.json').write_text(json.dumps(dict(wasm_sha256=sha,
            pack_hash=m['pack_hash'], cached_tensor=m['source_tensor'], cached_bytes=source_tensor['bytes'],
            scope='One real base Q projection only; no full-model or query-count improvement claim',
            extended_real_shapes=args.extended, baseline=args.baseline,
            inference_queries=3 * len(fixtures), weight_preparation_updates=3, status_queries=2, certified_module_reads=2,
            initial=initial, preparation=warm, idempotent_preparation=duplicate, clear=clear,
            final=final, phases=results,
            native_sha256=hashlib.sha256((ROOT / args.native).read_bytes()).hexdigest()), indent=2)+'\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
