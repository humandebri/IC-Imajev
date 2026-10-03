#!/usr/bin/env python3
"""Exercise late invalid INT8 lanes and a valid retry on a selected local module."""
import argparse
import hashlib
import json
import pathlib
import struct
import sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, decode, frame_digest
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--directory', required=True)
    a = ap.parse_args()
    source = ROOT / 'artifacts/delta-projected-v1-normal'
    report = json.loads((source / 'report.json').read_text())
    query = next(q for q in report['queries'] if q['op'] == 'mlp_gate_up_reuse')
    stem = f"{query['index']:06d}"
    raw = (source / 'queries' / f'{stem}.request.bin').read_bytes()
    h, _ = decode(raw)
    assert h['encoding'] == 'projection-block256-exact-v1'
    length, = struct.unpack('<I', raw[:4])
    payload = 4 + length
    assert raw[payload] == 2
    active = h['dims'][0] * h['dims'][2]
    sha = hashlib.sha256((ROOT / a.wasm).read_bytes()).hexdigest()
    d = ROOT / a.directory
    d.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    t = Transport(m['model'], 'http://localhost:8001/', a.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), d, m['pack_hash'])
    rejected = []
    try:
        verify_module(t, sha)
        for name, index in [('first', 0), ('last', active - 1)]:
            body = bytearray(raw[:-32])
            body[payload + 1 + index] = 128
            request = d / f'bad-{name}.bin'
            request.write_bytes(body + frame_digest(h, body))
            try:
                t.command(dict(op='step', input=str(request), output=str(d / f'bad-{name}.reply.bin')))
            except RuntimeError as e:
                assert 'projection codec integer' in str(e), str(e)
                rejected.append(dict(name=name, error=str(e)))
            else:
                raise AssertionError('Accepted reserved INT8 lane ' + name)
        request = d / 'valid.bin'
        request.write_bytes(raw)
        reply = d / 'valid.reply.bin'
        measured = t.command(dict(op='step', input=str(request), output=str(reply)))
        expected = decode((source / 'queries' / f'{stem}.response.bin').read_bytes())[1]
        got = decode(reply.read_bytes())[1]
        np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
        verify_module(t, sha)
        (d / 'report.json').write_text(json.dumps(dict(
            wasm_sha256=sha, request_sha256=hashlib.sha256(raw).hexdigest(),
            source_report_sha256=hashlib.sha256((source / 'report.json').read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            rejected=rejected, valid_retry_bitwise_equal=True, measurement=measured,
            rejected_ordinary_queries=2, successful_ordinary_queries=1,
            certified_module_reads=2), indent=2) + '\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
