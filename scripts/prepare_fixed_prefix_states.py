#!/usr/bin/env python3
"""Prepare 24 exact five-token Delta prefix states on a selected local canister."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from paid_prefix import load_prefix, COMMON_PREFIX
from prefix_inference import verify_module
from transport import Transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister', required=True)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True, help='Saved five-token prefix report and queries')
    parser.add_argument('--packets', type=Path, required=True, help='Matching validated NPF1 packet cache')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    cache = load_prefix(args.prefix, args.packets, manifest)
    # Validate every operand before any owner update.
    operands = []
    for layer, state in enumerate(cache['states']):
        if layer % 4 == 3:
            continue
        log = np.asarray(state['delta_log'], dtype='<f4').tobytes()
        if len(log) != len(COMMON_PREFIX) * 6176 * 4:
            raise ValueError('fixed prefix log shape')
        operands.append((layer, log, cache['hybrid_packets'][layer]))
    helper = ROOT / 'target/debug/examples/prefix_state_args'
    if not helper.is_file():
        raise RuntimeError('Run cargo build -p imajev-client --example prefix_state_args first')
    module = hashlib.sha256(args.wasm.read_bytes()).hexdigest()
    args.directory.mkdir(parents=True, exist_ok=False)
    did = args.directory / 'prepare.did'
    did.write_text('service:{prepareFixedPrefixCache:(blob,blob)->(variant {Ok:record {nat32;nat64;nat64};Err:text})}')
    transport = Transport(manifest['model'], 'http://localhost:8001/', args.canister,
                          str(ROOT / 'artifacts/imajev-local.pem'), args.directory / 'module', manifest['pack_hash'])
    rows = []
    try:
        verify_module(transport, module)
        for layer, log, packet in operands:
            log_path = args.directory / f'layer-{layer:02d}.log.f32'
            packet_path = args.directory / f'layer-{layer:02d}.npf1'
            argument = args.directory / f'layer-{layer:02d}.args.bin'
            reply = args.directory / f'layer-{layer:02d}.reply.hex'
            log_path.write_bytes(log)
            packet_path.write_bytes(packet)
            subprocess.run([str(helper), 'args', str(log_path), str(packet_path), str(argument)], check=True)
            started = time.monotonic()
            raw = subprocess.check_output(['icp', 'canister', 'call', args.canister, 'prepareFixedPrefixCache',
                '--network', 'local', '--identity', 'imajev-local', '--candid', str(did),
                '--args-file', str(argument), '--args-format', 'bin', '--output', 'hex'], text=True, cwd=ROOT)
            reply.write_text(raw)
            result = json.loads(subprocess.check_output([str(helper), 'decode', str(reply)], text=True))
            # Warm caches may already contain later entries; retries are idempotent.
            if not len(rows) + 1 <= result['count'] <= 24 or result['bytes'] != result['count'] * 2_097_152:
                raise ValueError('fixed prefix cache result')
            rows.append(dict(layer=layer, log_sha256=hashlib.sha256(log).hexdigest(),
                packet_sha256=hashlib.sha256(packet).hexdigest(), result=result, seconds=time.monotonic() - started))
            (args.directory / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        verify_module(transport, module)
    finally:
        transport.close()
    (args.directory / 'report.json').write_text(json.dumps(dict(canister=args.canister, network='local',
        wasm_sha256=module, prefix_tokens=5, update_calls=len(rows), layers=rows), indent=2) + '\n')
    print('Prepared 24 five-token Delta prefix states.')


if __name__ == '__main__':
    main()
