#!/usr/bin/env python3
"""Compare reused RoPE and fused RMS/RoPE to saved real ordinary-query outputs."""
import argparse
import hashlib
import json
import pathlib
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
    ap.add_argument('--baseline', default='delta-reuse-v1')
    a = ap.parse_args()
    dest = ROOT / a.directory
    dest.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    sha = hashlib.sha256((ROOT / a.wasm).read_bytes()).hexdigest()
    t = Transport(m['model'], 'http://localhost:8001/', a.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash'])
    seen, cases = set(), []
    try:
        verify_module(t, sha)
        for name in ['prefix', '617', 'insufficient', 'maximum', 'normal']:
            source = ROOT / f'artifacts/{a.baseline}-{name}'
            r = json.loads((source / 'report.json').read_text())
            headers = {q['index']: decode((source / 'queries' / f"{q['index']:06d}.request.bin").read_bytes())[0]
                       for q in r['queries']}
            for q in r['queries']:
                h = headers[q['index']]
                if h['op'] != 'rms_bf16' or not h['tensor'].endswith(('.self_attn.q_norm.weight', '.self_attn.k_norm.weight')):
                    continue
                heads = 16 if h['tensor'].endswith('.q_norm.weight') else 4
                n, width = h['dims'][0] // heads, h['dims'][1]
                rotation = next(p for p in r['queries'] if p['index'] > q['index'] and headers[p['index']]['op'] == 'rope_heads'
                                and headers[p['index']]['dims'][0] == n and headers[p['index']]['dims'][4] == heads)
                rh = headers[rotation['index']]
                key = (heads, tuple(rh['dims']))
                if key in seen:
                    continue
                seen.add(key)
                path = source / 'queries' / f"{q['index']:06d}.request.bin"
                values = decode(path.read_bytes())[1].reshape(n, heads, width).transpose(1, 0, 2)
                old_response = source / 'queries' / f"{rotation['index']:06d}.response.bin"
                expected = decode(old_response.read_bytes())[1]
                header = dict(rh, op='norm_rope_heads_bf16', tensor=h['tensor'], scalars=[h['scalars'][0], rh['scalars'][0]], aux=[])
                request = dest / f'{len(cases):03d}.request.bin'
                response = dest / f'{len(cases):03d}.response.bin'
                request.write_bytes(encode(header, values))
                result = t.command(dict(op='step', input=str(request), output=str(response)))
                got = decode(response.read_bytes())[1]
                assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), key
                # Also validate the legacy operation after its internal angle reuse.
                raw_request = source / 'queries' / f"{rotation['index']:06d}.request.bin"
                raw_response = dest / f'{len(cases):03d}.legacy.bin'
                raw = t.command(dict(op='step', input=str(raw_request), output=str(raw_response)))
                assert np.array_equal(decode(raw_response.read_bytes())[1].view(np.uint32), expected.view(np.uint32)), key
                case = dict(dims=rh['dims'], tensor=h['tensor'], bitwise_equal=True, legacy_bitwise_equal=True,
                            before_instructions=q['ok']['instructions']+rotation['ok']['instructions'],
                            before_candid_bytes=sum(v['ok']['request_bytes']+v['ok']['reply_bytes'] for v in (q,rotation)),
                            legacy=raw, **result)
                cases.append(case)
                print(json.dumps(case), flush=True)
        verify_module(t, sha)
        (dest / 'report.json').write_text(json.dumps(dict(cases=cases, wasm_sha256=sha,
            scope='Distinct real Q/K norms and positions; full graph validation required'), indent=2)+'\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
