#!/usr/bin/env python3
"""Export verified common27 states as immutable browser protocol assets."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from prefix_inference import load_cache


def main():
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    source = ROOT / 'artifacts/query-packing-v3/prefix-v2'
    report = json.loads((source / 'report.json').read_text())
    cache = load_cache(source / 'queries', manifest, report['wasm_sha256'])
    assert len(cache['metadata']['token_ids']) == 27 and len(cache['states']) == 32
    output = ROOT / 'frontend/public/inference/prefix27-v1'
    output.mkdir(parents=True, exist_ok=True)
    assets = {}
    for layer, state in enumerate(cache['states']):
        if layer % 4 == 3:
            values = np.concatenate([state['keys'].transpose(1, 0, 2).ravel(),
                                     state['values'].transpose(1, 0, 2).ravel()]).astype('<f4')
            assert values.size == 27 * 2048 and not np.any(values.view('<u4') & 65535)
            raw = (values.view('<u4') >> 16).astype('<u2').tobytes()
        else:
            log = np.asarray(state['delta_log'], dtype='<f4').ravel()
            conv = np.asarray(state['conv'], dtype='<f4').ravel()
            assert conv.size == 24576 and log.size == 27 * 6176
            bf = np.concatenate([conv, log[:27 * 2048]])
            assert not np.any(bf.view('<u4') & 65535)
            assert np.isfinite(log).all() and np.isfinite(conv).all()
            raw = (bf.view('<u4') >> 16).astype('<u2').tobytes() + log[27 * 2048:].tobytes()
        name = f'layer-{layer:02d}.bin'
        (output / name).write_bytes(raw)
        assets[str(layer)] = dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    data = dict(model=manifest['model'], pack_hash=manifest['pack_hash'],
                model_bytes=manifest['bytes'], prefix=cache['metadata']['token_ids'], assets=assets)
    encoded = (json.dumps(data, separators=(',', ':')) + '\n').encode()
    (output / 'manifest.json').write_bytes(encoded)
    release = dict(canister='xis3j-paaaa-aaaai-axumq-cai', host='https://icp-api.io',
                   module_hash='3291a064976fb13df548582871ae96a3347f052569eceb7b9807dc23ec5b2495',
                   model=manifest['model'], pack_hash=manifest['pack_hash'],
                   manifest_sha256=hashlib.sha256(encoded).hexdigest())
    (ROOT / 'frontend/src/inference-release.json').write_text(json.dumps(release, indent=2) + '\n')
    print(f'Exported 32 verified prefix assets ({sum(a["bytes"] for a in assets.values()):,} bytes).')


if __name__ == '__main__':
    main()
