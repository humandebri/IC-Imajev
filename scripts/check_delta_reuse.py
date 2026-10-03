#!/usr/bin/env python3
"""Check reused Delta operands against saved Wasm results and the old native path."""
import argparse
import hashlib
import json
import pathlib
import subprocess
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
    ap.add_argument('--directory', required=True)
    ap.add_argument('--baseline', default='block-codec-v1')
    ap.add_argument('--native', default='artifacts/strassen-wide/default-primitive')
    a = ap.parse_args()
    dest = ROOT / a.directory
    dest.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'checkpoints/full-int8.manifest.json'
    m = json.loads(manifest.read_text())
    sha = hashlib.sha256((ROOT / a.wasm).read_bytes()).hexdigest()
    t = Transport(m['model'], 'http://localhost:8001/', a.canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), dest, m['pack_hash'])
    seen, cases = set(), []
    try:
        verify_module(t, sha)
        for name in ['prefix', '617', 'insufficient', 'maximum', 'normal']:
            source = ROOT / f'artifacts/{a.baseline}-{name}'
            report = json.loads((source / 'report.json').read_text())
            for q in report['queries']:
                if q['op'] not in ('delta_stage_bf16', 'delta_input_integer', 'delta_gates_integer'):
                    continue
                request = source / 'queries' / f"{q['index']:06d}.request.bin"
                h, _ = decode(request.read_bytes())
                key = (h['op'], tuple(h['dims']))
                if key in seen:
                    continue
                seen.add(key)
                prefix = dest / f'{len(cases):03d}'
                response = prefix.with_suffix('.response.bin')
                result = t.command(dict(op='step', input=str(request), output=str(response)))
                got = decode(response.read_bytes())[1]
                old = decode((source / 'queries' / f"{q['index']:06d}.response.bin").read_bytes())[1]
                assert np.array_equal(got.view(np.uint32), old.view(np.uint32)), key
                native = prefix.with_suffix('.native.bin')
                subprocess.run([str(ROOT / a.native), str(request), str(native), str(manifest),
                                str(ROOT / 'checkpoints/full-int8.pack')], check=True)
                expected = decode(native.read_bytes())[1]
                native_equal = bool(np.array_equal(got.view(np.uint32), expected.view(np.uint32)))
                # Native exp/sigmoid may differ from Wasm by a few F32 ULPs.
                # Preserve the strict Wasm regression check above and separately
                # require exactness for the integer QKV projection itself.
                if h['op'] == 'delta_input_integer':
                    end = h['dims'][0] * 8192
                    assert np.array_equal(got[:end].view(np.uint32), expected[:end].view(np.uint32)), key
                case = dict(op=h['op'], dims=h['dims'], source=str(request.relative_to(ROOT)),
                            bitwise_equal=True, old_native_bitwise_equal=native_equal,
                            old_native_max_abs_error=float(np.max(np.abs(got-expected), initial=0)),
                            native_qkv_bitwise_equal=True if h['op'] == 'delta_input_integer' else None,
                            before_instructions=q['ok']['instructions'], **result)
                cases.append(case)
                print(json.dumps(case), flush=True)
        verify_module(t, sha)
        (dest / 'report.json').write_text(json.dumps(dict(cases=cases, wasm_sha256=sha,
            native_sha256=hashlib.sha256((ROOT / a.native).read_bytes()).hexdigest(),
            scope='Distinct real Delta shapes from five full graphs; separate full graph validation required'),
            indent=2) + '\n')
    finally:
        t.close()


if __name__ == '__main__':
    main()
