#!/usr/bin/env python3
"""Compare identical projection fixtures in disposable loopback IC canisters."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity', default='imajev-local')
    args = parser.parse_args()
    proof = json.loads(args.proof.read_text())
    modules = {'before': args.before, 'after': args.after}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert proof['complete'] and all(row['bitwiseEqual'] for row in proof['rows'])
    assert digest(args.before) == proof['baselineSHA256'] and digest(args.after) == proof['candidateSHA256']
    args.output.mkdir(parents=True, exist_ok=False)
    options = ['--network', 'local', '--identity', args.identity]
    def call(*command):
        return subprocess.check_output(['icp', *command], cwd=ROOT, text=True, stdin=subprocess.DEVNULL).strip()
    network = json.loads(call('network', 'status', '--json'))
    assert urlparse(network['api_url']).hostname in ('localhost', '127.0.0.1', '::1')
    report = {'scope': 'Synthetic quantization plus projection; IC performance_counter(0), not full-model inference',
              'proof_sha256': digest(args.proof), 'modules': {name: digest(path) for name,path in modules.items()},
              'api_url': network['api_url'], 'canisters': {}, 'rows': [], 'cleanup': {}, 'complete': False}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        for name, path in modules.items():
            principal = call('canister', 'create', '--detached', '--quiet', *options)
            assert re.fullmatch(r'[a-z2-7-]+-cai', principal), principal
            report['canisters'][name] = principal
            save()
            call('canister', 'install', principal, '--mode', 'install', '--wasm', str(path.resolve()), '--args', '()', *options)
        nonce = 0
        for row in proof['rows']:
            if row['pattern'] != 0 or row['tokens'] not in (1,69,87,132) or row['cols'] == 512:
                continue
            args_hex = struct.pack('<IIII', row['mode'], row['tokens'], row['cols'], row['pattern']).hex()
            measurements = {'before': [], 'after': []}
            for principal in report['canisters'].values():
                call('canister', 'call', principal, 'configure', args_hex, '--args-format', 'hex', '--output', 'hex', *options)
            for repeat in range(3):
                for name in (['before', 'after'] if repeat % 2 == 0 else ['after', 'before']):
                    nonce += 1
                    raw = bytes.fromhex(call('canister', 'call', report['canisters'][name], 'measure',
                        struct.pack('<I',nonce).hex(), '--query', '--args-format', 'hex', '--output', 'hex', *options).removeprefix('0x'))
                    instructions, output_hash, count, pages = struct.unpack('<QQII', raw)
                    assert count == row['tokens'] * 32
                    assert f'{output_hash:016x}' == row['outputFnv64'], 'IC output differs from Wasm/V8 proof'
                    measurements[name].append({'instructions':instructions, 'output_fnv64': f'{output_hash:016x}', 'heap_pages':pages})
            assert len({m['output_fnv64'] for samples in measurements.values() for m in samples}) == 1
            assert all(len({m['instructions'] for m in samples}) == 1 for samples in measurements.values())
            ratio = measurements['after'][0]['instructions'] / measurements['before'][0]['instructions']
            report['rows'].append({**{key:row[key] for key in ('mode','tokens','cols','pattern')}, 'measurements':measurements, 'instruction_ratio':ratio})
            save()
            print(json.dumps({'mode':row['mode'], 'tokens':row['tokens'], 'cols':row['cols'], 'ratio':ratio}), flush=True)
        report['complete'] = True
    finally:
        for name, principal in report['canisters'].items():
            report['cleanup'][name] = {}
            for action in ('stop','delete'):
                result = subprocess.run(['icp','canister',action,principal,*options], cwd=ROOT, text=True, capture_output=True, stdin=subprocess.DEVNULL)
                report['cleanup'][name][action] = {'returncode':result.returncode, 'stderr':result.stderr}
        save()

if __name__ == '__main__':
    main()
