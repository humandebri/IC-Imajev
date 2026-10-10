#!/usr/bin/env python3
"""Export verified common5 states as immutable browser protocol assets."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from prefix_inference import load_cache


def main():
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/browser-prefix5-v1')
    parser.add_argument('--module-hash')
    parser.add_argument('--write-release', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    report = json.loads((source / 'report.json').read_text())
    cache = load_cache(source / 'queries', manifest, report['wasm_sha256'])
    assert cache['metadata']['token_ids'] == [248045, 846, 198, 1349, 25] and len(cache['states']) == 32
    release_path = ROOT / 'frontend/src/inference-release.json'
    release = json.loads(release_path.read_text())
    module = args.module_hash or release['module_hash']
    if report['wasm_sha256'] != module:
        raise ValueError('cache/runtime release mismatch')
    output = ROOT / 'frontend/public/inference/prefix5-v1'
    output.mkdir(parents=True, exist_ok=True)
    assets = {}
    for layer, state in enumerate(cache['states']):
        if layer % 4 == 3:
            values = np.concatenate([state['keys'].transpose(1, 0, 2).ravel(),
                                     state['values'].transpose(1, 0, 2).ravel()]).astype('<f4')
            assert values.size == 5 * 2048 and not np.any(values.view('<u4') & 65535)
            raw = (values.view('<u4') >> 16).astype('<u2').tobytes()
        else:
            log = np.asarray(state['delta_log'], dtype='<f4').ravel()
            conv = np.asarray(state['conv'], dtype='<f4').ravel()
            assert conv.size == 24576 and log.size == 5 * 6176
            bf = np.concatenate([conv, log[:5 * 2048]])
            assert not np.any(bf.view('<u4') & 65535)
            assert np.isfinite(log).all() and np.isfinite(conv).all()
            raw = (bf.view('<u4') >> 16).astype('<u2').tobytes() + log[5 * 2048:].tobytes()
        name = f'layer-{layer:02d}.bin'
        (output / name).write_bytes(raw)
        assets[str(layer)] = dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    data = dict(model=manifest['model'], pack_hash=manifest['pack_hash'],
                model_bytes=manifest['bytes'], prefix=cache['metadata']['token_ids'], assets=assets,
                module_hash=module, prompt_layout='text-only-state-prefix5-v1')
    encoded = (json.dumps(data, separators=(',', ':')) + '\n').encode()
    (output / 'manifest.json').write_bytes(encoded)
    # Content-addressed URLs can be cached across reloads without stale-release risk.
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    immutable = ROOT / 'frontend/public/inference/immutable' / manifest_hash
    immutable.mkdir(parents=True, exist_ok=True)
    (immutable / 'manifest.json').write_bytes(encoded)
    for entry in assets.values():
        shutil.copyfile(output / entry['file'], immutable / entry['file'])
    if args.write_release:
        release.update(module_hash=module, manifest_sha256=manifest_hash,
                       prefix_tokens=5, max_tokens=96, prompt_layout='text-only-state-prefix5-v1')
        release_path.write_text(json.dumps(release, indent=2) + '\n')
    elif release['manifest_sha256'] != manifest_hash:
        raise ValueError('generated manifest differs from pinned release')
    print(f'Exported 32 verified prefix assets ({sum(a["bytes"] for a in assets.values()):,} bytes).')


if __name__ == '__main__':
    main()
