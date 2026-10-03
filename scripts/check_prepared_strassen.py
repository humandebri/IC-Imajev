#!/usr/bin/env python3
"""Measure exact coefficients prepared once; compare cold/warm/clear queries."""
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
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--source', default='artifacts/prepared-f32-v1-617')
    ap.add_argument('--native', default='artifacts/strassen-wide/default-primitive')
    args = ap.parse_args()
    dest = ROOT / args.directory
    dest.mkdir(parents=True, exist_ok=True)
    pack_dir = ROOT / 'artifacts/strassen-prepacked'
    m = json.loads((pack_dir / 'manifest.json').read_text())
    source = ROOT / args.source
    report = json.loads((source / 'report.json').read_text())
    tensor = m['source_tensor']
    query = next(q for q in report['queries'] if q['op'] == 'lora_integer' and
        decode((source / 'queries' / f"{q['index']:06d}.request.bin").read_bytes())[0]['tensor'] == tensor)
    source_request = source / 'queries' / f"{query['index']:06d}.request.bin"
    header, values = decode(source_request.read_bytes())
    values = values.reshape(header['dims'][0], header['dims'][2])
    # Other actual lengths come from the same layer in independently saved graphs.
    inputs = {n: values[:n] for n in [1, 7, 8, 32, 64, 87]}
    input_sources = {}
    for n, label in [(45, 'prefix'), (80, 'insufficient'), (89, 'maximum')]:
        directory = ROOT / f'artifacts/prepared-f32-v1-{label}'
        graph = json.loads((directory / 'report.json').read_text())
        q = next(q for q in graph['queries'] if q['op'] == 'lora_integer' and
            decode((directory / 'queries' / f"{q['index']:06d}.request.bin").read_bytes())[0]['tensor'] == tensor)
        p = directory / 'queries' / f"{q['index']:06d}.request.bin"
        h, x = decode(p.read_bytes())
        assert h['dims'][0] == n and h['dims'][2] == 2560
        inputs[n] = x.reshape(n, 2560)
        input_sources[str(n)] = dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    sha = hashlib.sha256((ROOT / args.wasm).read_bytes()).hexdigest()
    t = Transport(m['model'], 'http://localhost:8001/', args.canister,
        str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash'])
    cases, updates, oracles = [], [], {}
    try:
        verify_module(t, sha)
        pack = t.command(dict(op='pack_status'))['ok']
        assert pack['ready'] and pack['model'] == m['model'] and pack['pack_hash'] == m['pack_hash']
        assert pack['bytes'] == pack['received'] == pack['hashed'] == m['bytes']
        for phase in ['cold', 'warm', 'cleared']:
            if phase != 'warm':
                updates.append(t.command(dict(op='clear_weight_cache')))
                expected_cache = 0
            else:
                updates.append(t.command(dict(op='warm_weights', name=tensor)))
                prepared = t.command(dict(op='warm_weights', name=tensor + '.strassen_i8'))
                updates.append(prepared)
                expected_cache = 94_437_376
                assert prepared['ok']['cache']['bytes'] == expected_cache
                duplicate = t.command(dict(op='warm_weights', name=tensor + '.strassen_i8'))
                updates.append(duplicate)
                assert duplicate['ok']['cache']['bytes'] == expected_cache
            cache = t.command(dict(op='weight_cache_status'))['ok']['cache']
            assert cache['bytes'] == expected_cache
            assert set(cache['names']) == ({tensor, tensor + '.strassen_i8'} if phase == 'warm' else set())
            for n, x in sorted(inputs.items()):
                results, outputs = {}, {}
                for op in ['linear_integer_bf16', 'linear_strassen_bf16']:
                    h = dict(header, op=op, pack_hash=m['pack_hash'], dims=[n,8192,2560,0],
                        scalars=[], aux=[] if op == 'linear_integer_bf16' else [tensor+'.strassen_i8',tensor+'.strassen_corrections'], encoding='bf16-exact')
                    request = dest / f'{phase}-{n}-{op}.request.bin'
                    response = dest / f'{phase}-{n}-{op}.response.bin'
                    request.write_bytes(encode(h, x))
                    results[op] = t.command(dict(op='step', input=str(request), output=str(response)))
                    assert 'ok' in results[op], results[op]
                    outputs[op] = decode(response.read_bytes())[1]
                    if phase == 'warm': assert results[op]['ok']['stable_read_bytes'] == 0
                request = dest / f'{phase}-{n}-linear_integer_bf16.request.bin'
                request_hash = hashlib.sha256(request.read_bytes()).hexdigest()
                if n not in oracles:
                    native = dest / f'{n}.native.bin'
                    subprocess.run([str(ROOT / args.native), str(request), str(native), str(pack_dir/'manifest.json'), str(pack_dir/'pack.bin')], check=True)
                    oracles[n] = (request_hash, decode(native.read_bytes())[1], hashlib.sha256(native.read_bytes()).hexdigest())
                old_request_hash, expected, native_hash = oracles[n]
                assert request_hash == old_request_hash
                assert all(np.array_equal(v.view(np.uint32), expected.view(np.uint32)) for v in outputs.values())
                ordinary = results['linear_integer_bf16']['ok']['instructions']
                candidate = results['linear_strassen_bf16']['ok']['instructions']
                case = dict(phase=phase, tokens=n, bitwise_equal=True, native_bitwise_equal=True,
                    results=results, instruction_reduction=1-candidate/ordinary,
                    ordinary_request_sha256=request_hash, native_output_sha256=native_hash)
                cases.append(case)
                print(json.dumps(case), flush=True)
        verify_module(t, sha)
        (dest/'report.json').write_text(json.dumps(dict(wasm_sha256=sha, pack_hash=m['pack_hash'],
            cases=cases, updates=updates, source_request_sha256=hashlib.sha256(source_request.read_bytes()).hexdigest(),
            input_sources=input_sources, scope='Nine base Q shapes, cold/warm/clear; no full-model accuracy/query-count claim',
            inference_queries=len(cases)*2, native_invocations=len(oracles), cache_status_queries=3, pack_status_queries=1, certified_module_reads=2), indent=2)+'\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
