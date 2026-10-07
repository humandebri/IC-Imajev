#!/usr/bin/env python3
"""Read-only certification after restoring the dedicated local baseline."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport
from prefix_inference import verify_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('output must be a new directory')
    baseline = json.loads((ROOT / 'artifacts/local-goal-recovery-v1/baseline-state.json').read_text())
    before = json.loads((ROOT / 'artifacts/paid-common-raw-hybrid-v1/proof-storage/proof/before.json').read_text())
    assert baseline['module'] == before['module']
    assert baseline['cache'] == before['cache']
    keys = ['bytes', 'hashed', 'model', 'pack_hash', 'ready', 'received']
    assert all(baseline['pack'][key] == before['pack'][key] for key in keys)
    manifest = json.loads((ROOT / 'checkpoints/full-int8.manifest.json').read_text())
    out.mkdir(parents=True)
    transport = Transport(manifest['model'], 'http://localhost:8001/',
                          '4caro-hl777-77775-aaaba-cai', str(ROOT / 'artifacts/imajev-local.pem'),
                          out / 'queries', manifest['pack_hash'],
                          bridge_binary=str(ROOT / 'artifacts/query-packing-v3/build/imajev-client'))
    try:
        verify_module(transport, baseline['module'])
        actual = dict(module=baseline['module'],
                      cache=transport.command(dict(op='weight_cache_status'))['ok']['cache'],
                      pack=transport.command(dict(op='pack_status'))['ok'])
    finally:
        transport.close()
    (out / 'actual-state.json').write_text(json.dumps(actual, indent=2) + '\n')
    assert actual['cache'] == baseline['cache'], 'cache differs from portable baseline'
    assert all(actual['pack'][key] == baseline['pack'][key] for key in keys), 'critical pack fields differ'
    assert actual == baseline, 'full baseline state differs'
    (out / 'verified.json').write_text(json.dumps(dict(complete=True,
        target='4caro-hl777-77775-aaaba-cai', module=actual['module'],
        cache_equal=True, critical_pack_fields_equal=True, full_state_equal=True), indent=2) + '\n')
    print('local baseline module/cache/full pack state verified')


if __name__ == '__main__':
    main()
