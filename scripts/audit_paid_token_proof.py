#!/usr/bin/env python3
"""Recheck saved Candid replies and all numerical hashes without network calls."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def bits(v):
    return hashlib.sha256(np.asarray(v, dtype='<f4').tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--helper', type=Path, required=True)
    a = ap.parse_args()
    d = a.directory.resolve()
    report = json.loads((d / 'report.json').read_text())
    assert report['complete'] and report['baseline_restored'] and report['test_caller_stopped']
    fault = report.get('fault_recovery')
    if fault:
        assert fault['failed']['result']['Err']['Failed']['refund'] == 'Done'
        assert 0 <= fault['caller_balance_before'] - fault['caller_balance_after'] < 100_000_000
        assert fault['failed_duplicate_no_charge'] and fault['fresh_inference_equal']
    checked = 0
    for directory in ['calls', 'caller-balance']:
        for path in sorted((d / directory).glob('*.json')):
            if path.name.endswith('.input.json'):
                continue
            row = json.loads(path.read_text())
            if 'kind' not in row:
                continue
            reply = ROOT / row['reply_path']
            if row['relay']:
                forward = json.loads(subprocess.check_output([str(a.helper), 'decode', 'forward', str(reply)], text=True))
                assert forward == row['forward'], path
                if 'Err' in forward['response']:
                    assert row['result'] == {'transport_error': forward['response']['Err']}
                    checked += 1
                    continue
                inner = path.with_suffix('.inner.hex')
                assert bytes.fromhex(inner.read_text().strip().removeprefix('0x')) == bytes(forward['response']['Ok'])
                reply = inner
            actual = json.loads(subprocess.check_output([str(a.helper), 'decode', row['kind'], str(reply)], text=True))
            assert actual == row['result'], path
            checked += 1
    for comparison in report['query_comparisons']:
        n = comparison['tokens']
        debug = json.loads((d / f'debug-{n}.json').read_text())
        qd = d / f'query-reference-{n}'
        for layer in range(32):
            hidden = np.load(qd / f'layer-{layer:02d}.npy', allow_pickle=False)
            assert bits(hidden[27:] if layer < 31 else hidden) == debug['hidden_hashes'][layer]
            with np.load(qd / 'states' / f'layer-{layer:02d}.npz', allow_pickle=False) as z:
                state = np.concatenate([z['keys'][27:].ravel(), z['values'][27:].ravel()]) if layer % 4 == 3 else z['conv'].ravel()
                assert bits(state) == debug['state_hashes'][layer]
        assert bits(np.load(qd / 'final-hidden.npy', allow_pickle=False)) == bits(debug['final_hidden'])
        actual = json.loads((qd / 'decision.json').read_text())
        paid = next(c for c in report['cases'] if c['tokens'] == n)['call']['result']['Ok']['decision']
        assert {k:v for k,v in actual.items() if k!='instructions'} == {k:v for k,v in paid.items() if k!='instructions'}
    audit = dict(raw_replies_redecoded=checked, all_saved_arrays_and_decisions_rechecked=True,
                 query_lengths=[c['tokens'] for c in report['query_comparisons']], baseline_restored=True)
    (d / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
