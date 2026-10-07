#!/usr/bin/env python3
"""Register only the validated 27-token common prefix on an explicit local target."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from prefix_inference import load_cache, verify_module
from transport import Transport

PREFIX = [248045,846,198,56555,279,2420,5721,321,4087,279,3296,1608,279,10661,12521,13,3301,1132,279,3074,2904,1970,13,198,1349,25,328]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_layers(source, packets, manifest):
    report = json.loads((source / 'report.json').read_text())
    cache = load_cache(source / 'queries', manifest, report['wasm_sha256'])
    metadata = json.loads((packets / 'cache.json').read_text())
    if cache['metadata']['token_ids'] != PREFIX or report['tokens'] != 27:
        raise ValueError('expected the 27-token common prefix')
    if metadata['identity']['source_report_sha256'] != sha(source / 'report.json'):
        raise ValueError('packet source mismatch')
    layers = []
    for layer, state in enumerate(cache['states']):
        packet = b''
        if layer % 4 == 3:
            values = np.concatenate([state['keys'].transpose(1,0,2).ravel(), state['values'].transpose(1,0,2).ravel()])
        else:
            values = state['conv'].ravel()
            packet = (packets / f'layer-{layer:02d}.npf1').read_bytes()
            entry = metadata['packets'][str(layer)]
            if len(packet) != entry['bytes'] or hashlib.sha256(packet).hexdigest() != entry['sha256']:
                raise ValueError('packet hash mismatch')
            if len(packet) < 12 or packet[:4] != b'NPF1' or int.from_bytes(packet[4:8], 'little') != 27:
                raise ValueError('packet prefix mismatch')
        values = np.asarray(values, dtype='<f4')
        if not np.isfinite(values).all() or np.any(values.view('<u4') & 65535):
            raise ValueError('prefix precision')
        layers.append((values.tobytes(), packet))
    return layers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister', required=True)
    parser.add_argument('--wasm', required=True)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    source = ROOT / 'artifacts/query-packing-v3/prefix-v2'
    packets = ROOT / 'artifacts/query-packing-v3/packets-v2'
    layers = load_layers(source, packets, manifest)  # Validate every layer before any update.
    module = sha(ROOT / args.wasm)
    directory = ROOT / args.directory
    directory.mkdir(parents=True, exist_ok=False)
    transport = Transport(manifest['model'], 'http://localhost:8001/', args.canister,
        str(ROOT / 'artifacts/imajev-local.pem'), directory / 'module', manifest['pack_hash'],
        bridge_binary=str(ROOT / 'artifacts/query-packing-v3/build/imajev-client'))
    rows = []
    try:
        verify_module(transport, module)
        for layer, (values, packet) in enumerate(layers):
            path = directory / f'layer-{layer:02d}.f32'
            path.write_bytes(values)
            command = dict(op='update_prefix', layer=layer, values=str(path))
            if packet:
                path = directory / f'layer-{layer:02d}.npf1'
                path.write_bytes(packet)
                command['packet'] = str(path)
            reply = transport.command(command)
            rows.append(dict(layer=layer, values_sha256=hashlib.sha256(values).hexdigest(),
                packet_sha256=hashlib.sha256(packet).hexdigest(), reply=reply))
            (directory / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        verify_module(transport, module)
    finally:
        transport.close()
    (directory / 'report.json').write_text(json.dumps(dict(canister=args.canister,
        wasm_sha256=module, prefix_tokens=27, update_calls=len(rows), layers=rows), indent=2) + '\n')

if __name__ == '__main__':
    main()
