#!/usr/bin/env python3
"""Measure removable payload bytes in saved real queries, without claiming IC speedups."""
import argparse
import collections
import hashlib
import json
import pathlib
import sys
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='artifacts/prefix-hit-617')
    ap.add_argument('--output', default='docs/query-payload-exploration.json')
    args = ap.parse_args()
    source = ROOT / args.source
    report = json.loads((source / 'first-report.json').read_text())
    totals = collections.Counter()
    cases = []
    for q in report['queries']:
        op = q['op']
        if op not in ('delta_heads_bf16', 'attention_heads_bf16'):
            continue
        path = source / 'queries' / f"{q['index']:06d}"
        request = path.with_suffix('.request.bin').read_bytes()
        response = path.with_suffix('.response.bin').read_bytes()
        h, x = decode(request)
        rh, y = decode(response)
        entry = dict(index=q['index'], op=op)
        if op == 'delta_heads_bf16':
            n, dk, dv, heads = h['dims']
            if n != report.get('processed_tokens', report['tokens']):
                raise ValueError('Cannot omit state when a head spans multiple token chunks')
            keep = heads * n * dv
            entry['omitted_reply_state_frame_bytes'] = len(response) - len(encode(rh, y[:keep]))
            state = y[keep:].astype('<f4').tobytes()
            entry['state_f32_bytes'] = len(state)
            entry['state_zlib_bytes'] = len(zlib.compress(state))
            activ = n * (2 * dk + dv)
            tail = 2 * n + dk * dv
            tails = x[heads * activ:].reshape(heads, tail)
            if not np.any(tails[:, 2 * n:]):
                new_x = np.concatenate([x[:heads * activ], tails[:, :2 * n].ravel()])
                entry['omitted_zero_initial_state_frame_bytes'] = len(request) - len(encode(h, new_x))
        else:
            n, width, heads = h['dims']
            if heads % 4:
                raise ValueError('Exploration assumes complete aligned GQA groups of four')
            a = x.reshape(heads, 3, n, width)
            for start in range(0, heads, 4):
                for offset in range(1, 4):
                    assert np.array_equal(a[start, 1:].view(np.uint32), a[start + offset, 1:].view(np.uint32))
            unique = np.concatenate([a[:, 0].ravel(), a[::4, 1].ravel(), a[::4, 2].ravel()])
            entry['deduplicated_kv_frame_bytes'] = len(request) - len(encode(h, unique))
            prefix = report['tokens'] - report.get('processed_tokens', report['tokens'])
            entry['unused_prefix_score_pairs'] = heads * prefix * (prefix + 1) // 2
            entry['total_score_pairs'] = heads * n * (n + 1) // 2
        for key, value in entry.items():
            if key not in ('index', 'op'):
                totals[key] += value
        cases.append(entry)
    out = dict(scope='Offline exact frame encoding of real saved tensors; proposed compact frames are not current runtime requests; Candid framing and IC instructions are not measured; final Delta state omission applies only to terminal single-chunk inference, not prefix preparation or continuation', report_sha256=hashlib.sha256((source / 'first-report.json').read_bytes()).hexdigest(), source=args.source, original_candid_bytes=report['total_candid_bytes'], totals=dict(totals), cases=cases)
    (ROOT / args.output).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
