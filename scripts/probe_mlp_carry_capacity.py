#!/usr/bin/env python3
"""Measure lossless carry capacity using saved canister-produced operands.

This is a host byte-capacity probe. It does not implement a Wasm codec or prove
that merging MLP completion with the next Delta fits the instruction budget.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode


def planar(data, width):
    assert len(data) % width == 0
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, width).T.copy().tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='artifacts/mlp-pipeline-v1-617')
    ap.add_argument('--prefix', default='artifacts/prefix_codec/full-direct-input-proof/prefix')
    ap.add_argument('--output', default='artifacts/prefix_codec/mlp-carry-reproducible.json')
    args = ap.parse_args()
    source, prefix = ROOT / args.source, ROOT / args.prefix
    report_bytes = (source / 'report.json').read_bytes()
    report = json.loads(report_bytes)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    cases = []
    for query in report['queries']:
        if query['op'] != 'mlp_down_norm_prepared':
            continue
        layer = int(re.search(r'\.layers\.(\d+)\.', query['tensor']).group(1))
        if (layer + 1) % 4 == 3:
            continue  # The next layer is Attention, not Delta.
        request = (source / 'queries' / f'{query["index"]:06d}.request.bin').read_bytes()
        header, values = decode(request)
        n, cols = header['dims']
        assert cols == 2560 and header['encoding'] == 'mlp-down-state-exact-v1'
        c, q = n * cols, n * 9216
        assert values.size == c + q + n * 100
        residual_bits = values[:c].view('<u4')
        assert not np.any(residual_bits & 65535)
        residual = (residual_bits >> 16).astype('<u2').tobytes()
        quantized = values[c:c+q]
        assert np.array_equal(quantized, quantized.astype(np.int8).astype('<f4'))
        state_path = prefix / 'queries/states' / f'layer-{layer+1:02d}.npz'
        state_bytes = state_path.read_bytes()
        with np.load(state_path, allow_pickle=False) as state:
            log = state['delta_log']
            assert log.size == 45 * 6176
            log = log.reshape(-1)
            k = log[:45*2048].copy().view('<u4')
            assert not np.any(k & 65535)
            streams = [planar(residual, 2), quantized.astype(np.int8).tobytes(),
                       planar((k >> 16).astype('<u2').tobytes(), 2),
                       planar(log[45*2048:].astype('<f4').tobytes(), 4)]
            compressed = [zlib.compress(stream, 1) for stream in streams]
            assert all(zlib.decompress(packed) == raw for raw, packed in zip(streams, compressed))
            conv_bytes = state['conv'].size * 2
        # Exact scales/A, BF16 convolution history, bounded header, descriptors.
        overhead = n * 100 * 4 + conv_bytes + 16384 + 64
        cases.append(dict(layer=layer, request_sha256=sha(request), prefix_state_sha256=sha(state_bytes),
                          stream_raw_bytes=[len(s) for s in streams],
                          stream_compressed_bytes=[len(s) for s in compressed],
                          estimated_request_bytes=sum(map(len, compressed)) + overhead))
    assert len(cases) == 23
    result = dict(scope=__doc__, source_report_sha256=sha(report_bytes),
                  script_sha256=sha(pathlib.Path(__file__).read_bytes()), cases=cases)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), minimum_bytes=min(c['estimated_request_bytes'] for c in cases),
                          maximum_bytes=max(c['estimated_request_bytes'] for c in cases))))


if __name__ == '__main__':
    main()
