#!/usr/bin/env python3
"""Measure head groups with one shared quantization/A/gate preparation."""
import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, decode
from prefix_inference import verify_module

ap = argparse.ArgumentParser(description=__doc__)
for name in ['canister', 'wasm', 'directory']:
    ap.add_argument('--' + name, required=True)
a = ap.parse_args()
d = ROOT / a.directory
d.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
wasm = sha(ROOT / a.wasm)
m = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_bytes())
paths = list((ROOT / 'crates/imajev-runtime/src').rglob('*.rs')) + list((ROOT / 'canisters/inference/src').rglob('*.rs')) + list((ROOT / 'client').glob('*.py')) + [pathlib.Path(__file__), ROOT / 'MODEL_LOCK.json', ROOT / 'checkpoints/full-int8.manifest.json']
hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
t = Transport(m['model'], 'http://localhost:8001/', a.canister,
              str(ROOT / 'artifacts/imajev-local.pem'), d, m['pack_hash'],
              wire_codec='bf16-block256-exact-v1', frame_version=3)
cases, profiles = [], []
try:
    verify_module(t, wasm)
    for label in ['prefix', '617', 'insufficient', 'maximum']:
        source = ROOT / f'artifacts/column16-v1-{label}'
        report = json.loads((source / 'report.json').read_bytes())
        assert report['model'] == m['model'] and report['pack_hash'] == m['pack_hash']
        for layer in [0, 1, 30]:
            tensor = f'model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight'
            original = [q for q in report['queries'] if q['tensor'] == tensor and q['op'] in ('delta_project_capture', 'delta_project_reuse')]
            assert len(original) == 2
            requests, responses, sources = [], [], []
            for q in original:
                req = source / 'queries' / f"{q['index']:06d}.request.bin"
                reply = source / 'queries' / f"{q['index']:06d}.response.bin"
                requests.append(decode(req.read_bytes()))
                responses.append(decode(reply.read_bytes()))
                sources.append(dict(request_sha256=sha(req), response_sha256=sha(reply)))
            h, x = requests[0]
            n = h['dims'][0]
            assert h['dims'][1:3] == [16, 0] and requests[1][0]['dims'][1:3] == [16, 16]
            projection = next(q for q in report['queries'] if q['tensor'] == tensor.replace('in_proj_qkv', 'out_proj') and q['op'] == 'lora_integer')
            path = source / 'queries' / f"{projection['index']:06d}.request.bin"
            _, expected = decode(path.read_bytes())
            histories, states, final_histories = [], [], []
            for (rh, rx), (_, ry) in zip(requests, responses):
                offset = n * (2560 if rh['op'] == 'delta_project_capture' else 2762)
                histories.append(rx[offset:offset + 3 * 4096].reshape(3, 4096))
                states.append(rx[offset + 3 * 4096:offset + 3 * 4096 + 16 * 16384].reshape(16, 16384))
                final_histories.append(ry[n * 16 * 128:n * 16 * 128 + 3 * 4096].reshape(3, 4096))
            prepared = None
            gated, histories_match, calls, frames = [], [], [], []
            # The first 22 heads consist of 16+6 and share one preparation;
            # remaining 10 heads reuse it too. These are diagnostic queries,
            # not a claim that two layers already fit in three inference calls.
            for first, heads in [(0, 16), (16, 6), (22, 10)]:
                group, local = first // 16, first % 16
                def sliced_history(v):
                    return np.concatenate([v[:, local // 2 * 128:(local + heads) // 2 * 128],
                                           v[:, 1024 + local // 2 * 128:1024 + (local + heads) // 2 * 128],
                                           v[:, 2048 + local * 128:2048 + (local + heads) * 128]], axis=1).ravel()
                prefix = x[:n * 2560] if prepared is None else prepared
                values = np.concatenate([prefix, sliced_history(histories[group]), states[group][local:local + heads].ravel()])
                op = 'delta_project_capture' if prepared is None else 'delta_project_reuse'
                y = t.run(op, values, [n, heads, first, 0], tensor=tensor, input_hash=h['input_hash'])
                frames.append(t.index - 1)
                calls.append(t.measurements[-1])
                gated.append(y[:n * heads * 128].reshape(n, heads * 128))
                histories_match.append(y[n * heads * 128:n * heads * 128 + 3 * heads * 256].tobytes() == sliced_history(final_histories[group]).tobytes())
                if prepared is None:
                    prepared = y[n * heads * 128 + 3 * heads * 256:]
                    assert len(prepared) == n * 2762
            assert np.concatenate(gated, axis=1).ravel().tobytes() == expected.tobytes()
            assert all(histories_match)
            if label == '617' and layer == 0:
                for index in frames:
                    out = d / f'{index:06d}.profile.response.bin'
                    metric = t.command(dict(op='profile', input=str(d / f'{index:06d}.request.bin'), output=str(out)))
                    assert decode(out.read_bytes())[1].tobytes() == decode((d / f'{index:06d}.response.bin').read_bytes())[1].tobytes()
                    profiles.append(metric)
            case = dict(label=label, layer=layer, tokens=n, groups=[16, 6, 10], gated_bitwise_equal=True,
                        history_bitwise_equal=True, calls=calls, source=sources,
                        gated_reference_request_sha256=sha(path),
                        instructions=sum(c['ok']['instructions'] for c in calls),
                        candid_bytes=sum(c['ok']['request_bytes'] + c['ok']['reply_bytes'] for c in calls))
            cases.append(case)
            print(json.dumps(case), flush=True)
    verify_module(t, wasm)
    assert all(sha(ROOT / p) == v for p, v in hashes.items())
    result = dict(scope=__doc__, wasm_sha256=wasm, source_hashes=hashes, cases=cases, profiles=profiles,
                  regular_diagnostic_queries=len(t.measurements), profile_diagnostic_queries=len(profiles),
                  goal_50_verified=False)
    (d / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    with zipfile.ZipFile(d / 'validated-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p, str(p.relative_to(ROOT)))
finally:
    t.close()
