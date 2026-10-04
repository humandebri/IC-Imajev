#!/usr/bin/env python3
"""Measure exact-zero INT8 product blocks in saved validated MLP stream replies."""
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode

cases = []
for label in ['617', 'insufficient', 'maximum']:
    directory = ROOT / f'artifacts/f32_k_continue/stream-check-{label}'
    proof = json.loads((directory / 'report.json').read_bytes())
    assert all(c['bitwise_prepared_equal'] and c['bitwise_hidden_norm_equal'] for c in proof['cases'])
    seen = set()
    for path in sorted(directory.glob('*.response.bin')):
        header, values = decode(path.read_bytes())
        if header['op'] != 'mlp_stream_next' or sum(header['dims'][1:]) != 9216:
            continue
        tensor = header['tensor']
        if tensor in seen:
            continue
        seen.add(tensor)
        n = header['dims'][0]
        start = n * (2560 * 2 + 10 + 128)
        integers = values[start:start + n * 9216].reshape(n, 36, 256)
        assert np.all(integers == np.trunc(integers))
        zero = np.all(integers == 0, axis=2)
        cases.append(dict(label=label, tensor=tensor, tokens=n,
                          zero_lane_fraction=float(np.mean(integers == 0)),
                          exact_zero_token_blocks=int(zero.sum()), token_blocks=n * 36,
                          response_path=str(path.relative_to(ROOT)),
                          response_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          proof_sha256=hashlib.sha256((directory / 'report.json').read_bytes()).hexdigest()))
    assert len(seen) == 3
report = dict(scope=__doc__, cases=cases, optimization_applied=False)
(ROOT / 'artifacts/f32_k_continue/zero-block-exploration.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(cases, indent=2))
