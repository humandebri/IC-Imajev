#!/usr/bin/env python3
"""Make same-length four-message hash fixtures from verified historical states."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    best = ROOT / 'artifacts/paid-stack-carry-projection-v1'
    pointer = json.loads((best / 'validated-proof-pointer.json').read_text())
    assert pointer['complete'] and pointer['validated_best']
    for path, value in pointer['hashes'].items():
        assert sha(ROOT / path) == value, path
    proof = json.loads((best / 'proof/report.json').read_text())
    assert proof['complete'] and proof['baseline_restored']
    d = ROOT / 'artifacts/sha256-batch-real-fixtures-v1'
    d.mkdir(exist_ok=False)
    files = [Path(__file__), best / 'validated-proof-pointer.json', best / 'proof/report.json']
    groups = []
    for item in proof['results']:
        name = item['case']
        prefix = item['quote']['prefix_tokens']
        historical = ROOT / f'artifacts/boomdao-current-v1/{name}-r1/queries'
        for kind, layers in [('hidden', [0, 1, 2, 3]), ('conv', [0, 1, 2, 4]), ('kv', [3, 7, 11, 15])]:
            messages = []
            for layer in layers:
                if kind == 'hidden':
                    source = historical / f'layer-{layer:02d}.npy'
                    values = np.load(source, allow_pickle=False)[prefix:]
                    expected = item['debug']['hidden_hashes'][layer]
                else:
                    source = historical / 'states' / f'layer-{layer:02d}.npz'
                    with np.load(source, allow_pickle=False) as state:
                        if kind == 'conv':
                            values = state['conv'].copy()
                        else:
                            values = np.concatenate([state['keys'][prefix:].ravel(), state['values'][prefix:].ravel()])
                    expected = item['debug']['state_hashes'][layer]
                raw = np.asarray(values, dtype='<f4').tobytes()
                digest = hashlib.sha256(raw).hexdigest()
                assert digest == expected, (name, kind, layer)
                output = d / f'{name}-{kind}-{layer:02d}.bin'
                output.write_bytes(raw)
                files.extend([source, output])
                messages.append(dict(layer=layer, path=str(output.relative_to(ROOT)), bytes=len(raw), sha256=digest, source=str(source.relative_to(ROOT))))
            assert len({m['bytes'] for m in messages}) == 1
            groups.append(dict(case=name, kind=kind, bytes_per_message=messages[0]['bytes'], messages=messages))
    assert len(groups) == 9 and sum(len(g['messages']) for g in groups) == 36
    result = dict(complete=True, groups=groups, all36_hashes_equal_latest_verified_debug=True,
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(files)},
                  scope='Historical float bits independently matched to latest verified debug digests; nine equal-length four-message groups. No new current intermediate or Dense F32 capture. No SIMD code, IC timing, inference savings or adoption claim.')
    (d / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(groups=len(groups), messages=36, all_hashes_exact=True)))


if __name__ == '__main__':
    main()
