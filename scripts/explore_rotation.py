#!/usr/bin/env python3
"""Screen block-Hadamard W8/A8 error on real inputs, not end-to-end accuracy.

Requantizes original BF16 weights, retains current row weight and block256
activation scaling, and measures base projection only. No model is installed.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode
from checkpoint import Checkpoint


def rotate(a, signs):
    y = (a.reshape(len(a), -1, 256) * signs).copy()
    for width in (1, 2, 4, 8, 16, 32, 64, 128):
        z = y.reshape(*y.shape[:-1], -1, 2, width)
        left, right = z[..., 0, :].copy(), z[..., 1, :].copy()
        z[..., 0, :], z[..., 1, :] = left + right, left - right
    return y.reshape(a.shape) / np.float32(16)


def quantized_projection(x, w, block_size=256):
    sw = np.maximum(np.max(abs(w), axis=1) / np.float32(127), np.finfo(np.float32).tiny)
    qw = np.clip(np.rint(w / sw[:, None]), -127, 127).astype(np.int32)
    if x.shape[1] * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError('I32 accumulation bound')
    xb = x.reshape(len(x), -1, block_size)
    sx = np.max(abs(xb), axis=2) / np.float32(127)
    sx = np.where(sx == 0, np.float32(1), np.maximum(sx, np.nextafter(np.float32(0), np.float32(1))))
    qx = np.clip(np.rint(xb / sx[..., None]), -127, 127).astype(np.int32)
    y = np.zeros((len(x), len(w)), np.float32)
    for b in range(x.shape[1] // block_size):
        dot = qx[:, b] @ qw[:, b * block_size:(b + 1) * block_size].T
        y += (dot.astype(np.float32) * sx[:, b, None]) * sw[None, :]
    return y


def weight_only_projection(x, w):
    scale = np.maximum(np.max(abs(w), axis=1) / np.float32(127), np.finfo(np.float32).tiny)
    qw = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return x.astype(np.float64) @ (qw.astype(np.float32) * scale[:, None]).astype(np.float64).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='artifacts/prefix-hit-617')
    ap.add_argument('--output', default='docs/rotation-screen.json')
    args = ap.parse_args()
    source = ROOT / args.source
    report = json.loads((source / 'first-report.json').read_text())
    index = json.loads((ROOT / 'checkpoints/base/model.safetensors.index.json').read_text())['weight_map']
    selected = {}
    for q in report['queries']:
        if q['op'] not in ('lora_integer', 'mlp_gate_up_integer'):
            continue
        layer, kind = q['tensor'].split('.layers.')[1].split('.', 1)
        if int(layer) not in (0, 3, 12, 15, 24, 27, 31):
            continue
        key = (layer, kind)
        if key not in selected:
            selected[key] = q
            if q['op'] == 'mlp_gate_up_integer':
                selected[layer, kind.replace('gate_proj', 'up_proj')] = dict(q, tensor=q['tensor'].replace('gate_proj', 'up_proj'))
    cases = []
    readers = {}
    for key, q in selected.items():
        request = source / 'queries' / f"{q['index']:06d}.request.bin"
        h, values = decode(request.read_bytes())
        n, rows, cols, start = h['dims']
        # Spread sampled rows over the actual output tile; token rows include its tail.
        row_ids = start + np.unique(np.linspace(0, rows - 1, min(rows, 32), dtype=int))
        token_ids = np.unique(np.linspace(0, n - 1, min(n, 16), dtype=int))
        x = values.reshape(n, cols)[token_ids]
        shard = index[q['tensor']]
        if shard not in readers:
            readers[shard] = Checkpoint(ROOT / 'checkpoints/base' / shard)
        ck = readers[shard]
        info = ck.header[q['tensor']]
        if info['dtype'] != 'BF16':
            raise ValueError('expected original BF16 dense weight')
        raw = np.frombuffer(ck.map, dtype='<u2', count=int(np.prod(info['shape'])), offset=ck.offset + info['data_offsets'][0]).reshape(info['shape'])
        w = (raw[row_ids].astype(np.uint32) << 16).view(np.float32)
        reference = x.astype(np.float64) @ w.astype(np.float64).T
        def error(y):
            diff = y - reference
            return dict(rmse=float(np.sqrt(np.mean(diff ** 2))), max_abs=float(np.max(abs(diff))))
        baseline = error(quantized_projection(x, w))
        weight_only = error(weight_only_projection(x, w))
        variants = []
        for seed in (0, 1, 2):
            signs = np.random.default_rng(seed).choice(np.array([-1, 1], np.float32), size=(cols // 256, 256))
            rx, rw = rotate(x, signs), rotate(w, signs)
            result = error(quantized_projection(rx, rw))
            invariant = error(rx.astype(np.float64) @ rw.astype(np.float64).T)
            row_scale = error(quantized_projection(rx, rw, block_size=cols))
            rotated_weight_only = error(weight_only_projection(rx, rw))
            variants.append(dict(seed=seed, **result, rmse_ratio=result['rmse'] / baseline['rmse'], unquantized_rotation_error=invariant, weight_only=rotated_weight_only, weight_only_rmse_ratio=rotated_weight_only['rmse'] / weight_only['rmse'], activation_per_row=row_scale, activation_per_row_rmse_ratio=row_scale['rmse'] / baseline['rmse']))
        cases.append(dict(tensor=q['tensor'], input_sha256=hashlib.sha256(request.read_bytes()).hexdigest(), tokens=len(x), rows=len(w), cols=cols, baseline=baseline, baseline_weight_only=weight_only, baseline_activation_per_row=error(quantized_projection(x, w, block_size=cols)), rotations=variants))
        print(key, [round(v['rmse_ratio'], 3) for v in variants], flush=True)
    out = dict(scope='Sampled real base projections only; original BF16 weights and frozen baseline activations; no LoRA/readout/nonlinear propagation or judgment accuracy evaluation; no Wasm cost claim', model=report['model'], pack_hash=report['pack_hash'], seed_selection='Three fixed seeds, no optimization on evaluation labels', cases=cases)
    (ROOT / args.output).write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
