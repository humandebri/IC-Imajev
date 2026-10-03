#!/usr/bin/env python3
"""Profile real saved graph requests on an explicitly selected local module.

This script never installs modules. Inclusive spans must not be summed together;
instrumented normal queries are not substituted for production measurements.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, decode
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--directory', required=True)
    a = ap.parse_args()
    source, dest = ROOT / a.source, ROOT / a.directory
    dest.mkdir(parents=True, exist_ok=True)
    report = json.loads((source / 'report.json').read_text())
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    sha = hashlib.sha256((ROOT / a.wasm).read_bytes()).hexdigest()
    t = Transport(manifest['model'], 'http://localhost:8001/', a.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, manifest['pack_hash'])
    seen, cases = set(), []
    try:
        verify_module(t, sha)
        for q in report['queries']:
            request = source / 'queries' / f"{q['index']:06d}.request.bin"
            header, _ = decode(request.read_bytes())
            key = (header['op'], tuple(header['dims'][:3]))
            if key in seen:
                continue
            seen.add(key)
            expected = decode((source / 'queries' / f"{q['index']:06d}.response.bin").read_bytes())[1]
            measurements = {}
            for op in ('step', 'profile'):
                response = dest / f"{q['index']:06d}.{op}.bin"
                result = t.command(dict(op=op, input=str(request), output=str(response)))
                if 'ok' not in result:
                    raise RuntimeError(result)
                got = decode(response.read_bytes())[1]
                assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), key
                measurements[op] = result
            case = dict(op=header['op'], dims=header['dims'], tensor=header['tensor'],
                        production_instructions=q['ok']['instructions'],
                        request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
                        bitwise_equal=True, results=measurements)
            cases.append(case)
            print(json.dumps(case), flush=True)
        verify_module(t, sha)
        (dest / 'report.json').write_text(json.dumps(dict(
            cases=cases, wasm_sha256=sha, source_report_sha256=hashlib.sha256(
                (source / 'report.json').read_bytes()).hexdigest(),
            scope='Real saved requests; inclusive instrumented spans; separate production counters; not a full graph speed claim'),
            indent=2) + '\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
