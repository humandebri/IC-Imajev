#!/usr/bin/env python3
"""Exercise saved merged INT8 weights through the existing generic Rust kernel.

This checks real block256 inputs and nonzero output-row offsets against an
independent NumPy integer-dot oracle. It is not an end-to-end Wasm measurement.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode, encode


def bf(x):
    bits = np.asarray(x, dtype='<f4').view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', default='artifacts/merged-adapter-v2')
    args = ap.parse_args()
    directory = ROOT / args.directory
    manifest_path = directory / 'model.manifest.json'
    pack_path = directory / 'model.pack'
    manifest = json.loads(manifest_path.read_text())
    h, x = decode((ROOT / 'artifacts/integer-kernel/request.bin').read_bytes())
    tensor = next(t for t in manifest['tensors'] if t['name'] == h['tensor'])
    n, rows, start, cols = 4, 64, 8, tensor['cols']
    values = x.reshape(-1, cols)[:n]
    header = {**h, 'pack_hash': manifest['pack_hash'], 'op': 'linear_integer_bf16', 'dims': [n, rows, cols, start], 'aux': [], 'scalars': []}
    request, reply = directory / 'native.request.bin', directory / 'native.response.bin'
    request.write_bytes(encode(header, values))
    subprocess.run([str(ROOT / 'target/release/primitive'), str(request), str(reply), str(manifest_path), str(pack_path)], check=True)
    _, got = decode(reply.read_bytes())
    weights = np.memmap(pack_path, mode='r', dtype=np.int8, offset=tensor['offset'], shape=(tensor['rows'], cols))[start:start+rows]
    sw = np.memmap(pack_path, mode='r', dtype='<f4', offset=tensor['offset']+tensor['rows']*cols, shape=(tensor['rows'],))[start:start+rows]
    expected = np.zeros((n, rows), dtype=np.float32)
    for begin in range(0, cols, 256):
        block = values[:, begin:begin+256]
        peak = np.max(np.abs(block), axis=1)
        sx = np.where(peak==0, np.float32(1), np.maximum(peak/np.float32(127), np.nextafter(np.float32(0), np.float32(1)))).astype(np.float32)
        q = np.clip(np.rint(block/sx[:, None]), -127, 127).astype(np.int32)
        dot = q @ weights[:, begin:begin+256].astype(np.int32).T
        scaled = dot.astype(np.float32)*sx[:, None]
        scaled = scaled*sw[None, :]
        expected = expected+scaled
    expected = bf(expected)
    got = got.reshape(n, rows)
    equal = np.array_equal(got.view(np.uint32), expected.view(np.uint32))
    report = dict(scope='Existing native Rust generic INT8 projection, one real merged layer-0 tensor; no adapter, nonzero row offset. Not a Wasm/full-model result.',
                  tensor=tensor['name'], tokens=n, rows=rows, row_offset=start, pack_hash=manifest['pack_hash'],
                  bitwise_equal=bool(equal), max_abs_error=float(np.max(np.abs(got-expected))))
    (directory / 'native-check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if not equal:raise AssertionError(report)


if __name__ == '__main__':
    main()
