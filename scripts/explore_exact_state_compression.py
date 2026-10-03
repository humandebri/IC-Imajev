#!/usr/bin/env python3
"""Host-only lossless state size exploration; never changes inference or installs."""
import argparse
import hashlib
import json
import pathlib
import zlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    source = ROOT / args.source
    cache_bytes = (source / 'cache.json').read_bytes()
    cache = json.loads(cache_bytes)
    cases = []
    for layer in range(32):
        path = source / f'states/layer-{layer:02d}.npz'
        with np.load(path, allow_pickle=False) as data:
            if 'delta' not in data.files:
                continue
            state = data['delta']
        if state.shape != (32, 128, 128) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError('Expected exact F32 Delta state')
        bits = state.astype('<f4', copy=False).view('<u4').copy()
        raw = bits.tobytes()
        modes = {}
        for mode in ['raw', 'byte-plane', 'xor-column-byte-plane']:
            transformed = bits.copy()
            if mode.startswith('xor'):
                transformed[:, :, 1:] = bits[:, :, 1:] ^ bits[:, :, :-1]
            payload = transformed.tobytes()
            if mode != 'raw':
                payload = np.frombuffer(payload, np.uint8).reshape(-1, 4).T.copy().tobytes()
            compressed = zlib.compress(payload, 1)
            restored = zlib.decompress(compressed)
            if mode != 'raw':
                restored = np.frombuffer(restored, np.uint8).reshape(4, -1).T.copy().tobytes()
            recovered = np.frombuffer(restored, '<u4').reshape(bits.shape).copy()
            if mode.startswith('xor'):
                recovered = np.bitwise_xor.accumulate(recovered, axis=-1)
            assert recovered.tobytes() == raw
            modes[mode] = dict(compressed_bytes=len(compressed), ratio=len(compressed)/len(raw), bitwise_roundtrip=True)
        cases.append(dict(layer=layer, raw_bytes=len(raw), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), modes=modes))
    result = dict(scope='Host-only serialized F32 state sizes; no canister codec, command count, speed or accuracy improvement claim',
        source=args.source, cache_sha256=hashlib.sha256(cache_bytes).hexdigest(), model=cache.get('model'), pack_hash=cache.get('pack_hash'),
        algorithm='zlib level1; byte transforms reversible; no lossy quantization', cases=cases)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(layers=len(cases), raw_bytes=sum(c['raw_bytes'] for c in cases),
        compressed_bytes={mode:sum(c['modes'][mode]['compressed_bytes'] for c in cases) for mode in ['raw','byte-plane','xor-column-byte-plane']})))
if __name__ == '__main__':
    main()
