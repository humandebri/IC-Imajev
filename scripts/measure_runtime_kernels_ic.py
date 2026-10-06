#!/usr/bin/env python3
"""Measure already-validated synthetic kernels in two disposable local canisters.

Creates, installs, configures, queries, stops and deletes only the two canisters
created by this run. Requires a running loopback IC network and pinned build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--identity', default='imajev-local')
    args = ap.parse_args()
    build = (ROOT / args.build).resolve()
    dest = (ROOT / args.directory).resolve()
    if not build.is_relative_to(ROOT) or not dest.is_relative_to(ROOT):
        raise ValueError('paths must stay in repository')
    proof = json.loads((build / 'report.json').read_text())
    modules = {'imajev':build / 'imajev-patch-2.wasm', 'ggml':build / 'ggml.wasm'}
    for name, path in modules.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == proof['wasmHashes'][name]
    dest.mkdir(parents=True, exist_ok=False)
    cli = ['--network', 'local', '--identity', args.identity]

    def call(command):
        return subprocess.check_output(['icp', *command], cwd=ROOT, text=True, stdin=subprocess.DEVNULL).strip()

    network = json.loads(call(['network', 'status', '--json']))
    assert urlparse(network['api_url']).hostname in ('localhost', '127.0.0.1', '::1'), 'local loopback network required'
    report = {'scope':'Prepared synthetic projection, same local IC, handler performance_counter(0); not full-model performance',
              'api_url':network['api_url'], 'node_proof_sha256':hashlib.sha256((build/'report.json').read_bytes()).hexdigest(),
              'wasm_hashes':proof['wasmHashes'], 'measurement_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'canisters':{}, 'rows':[], 'cleanup':{}, 'error':None}
    try:
        for name, wasm in modules.items():
            principal = call(['canister', 'create', '--detached', '--quiet', *cli])
            assert re.fullmatch(r'[a-z2-7-]+', principal) and principal.endswith('-cai'), principal
            report['canisters'][name] = principal
            (dest / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
            call(['canister', 'install', principal, '--mode', 'install', '--wasm', str(wasm), '--args', '()', *cli])
        nonce = 0
        for case in proof['rows']:
            n, rows, cols = case['tokens'], case['rows'], case['cols']
            payload = struct.pack('<III', n, rows, cols).hex()
            preparation = {}
            for name, principal in report['canisters'].items():
                begin = time.monotonic()
                call(['canister', 'call', principal, 'configure', payload, '--args-format', 'hex', '--output', 'hex', *cli])
                preparation[name] = time.monotonic()-begin
            measurements = {'imajev':[], 'ggml':[]}
            for repeat in range(3):
                for name in (['imajev','ggml'] if repeat % 2 == 0 else ['ggml','imajev']):
                    nonce += 1
                    # Distinct query arguments prevent reuse of cached responses.
                    output = call(['canister', 'call', report['canisters'][name], 'measure', struct.pack('<I', nonce).hex(), '--query', '--args-format', 'hex', '--output', 'hex', *cli])
                    raw = bytes.fromhex(output.removeprefix('0x'))
                    assert len(raw) == 24, output
                    instructions, digest, count, pages = struct.unpack('<QQII', raw)
                    assert count == n * rows and f'{digest:016x}' == case['outputFnv64Hex'], (name, n, cols, digest)
                    measurements[name].append({'instructions':instructions, 'output_fnv64':f'{digest:016x}', 'output_values':count, 'heap_pages':pages, 'nonce':nonce})
            for name in measurements:
                assert len({m['instructions'] for m in measurements[name]}) == 1, (name, measurements[name])
            counts = {name:measurements[name][0]['instructions'] for name in measurements}
            row = {'tokens':n, 'rows':rows, 'cols':cols, 'measurements':measurements, 'instructions':counts,
                   'imajev_over_ggml':counts['imajev']/counts['ggml'], 'preparation_call_seconds':preparation,
                   'node_bitwise_scalar_proof':True}
            report['rows'].append(row)
            (dest / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'tokens':n, 'cols':cols, 'instructions':counts, 'ratio':row['imajev_over_ggml']}), flush=True)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        for name, principal in report['canisters'].items():
            cleanup = {}
            for action in ['stop', 'delete']:
                result = subprocess.run(['icp', 'canister', action, principal, *cli], cwd=ROOT, text=True, stdin=subprocess.DEVNULL, capture_output=True)
                cleanup[action] = {'exit_code':result.returncode, 'stdout':result.stdout, 'stderr':result.stderr}
            report['cleanup'][name] = cleanup
        (dest / 'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
