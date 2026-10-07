#!/usr/bin/env python3
"""Recheck all saved worker replies and references before comparing full-update costs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bits(v):
    return hashlib.sha256(np.asarray(v, dtype='<f4').tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    d = ROOT / args.directory
    proof_path = d / 'proof/report.json'
    proof = json.loads(proof_path.read_text())
    build = json.loads((d / 'build/report.json').read_text())
    workflow = d / 'workflow-hashes.json'
    if workflow.exists():
        assert all(sha(ROOT / p) == h for p,h in json.loads(workflow.read_text()).items())
    assert proof['complete'] and proof['baseline_snapshot_restored'] and proof['snapshot_deleted']
    assert sha(d / 'build/full.wasm') == proof['candidate'] == build['wasm_sha256']
    for manifest in [proof['source_hashes'], build['source_hashes'], build['dependency_hashes']]:
        assert all(sha(ROOT / p) == h for p, h in manifest.items())
    restored = json.loads((d / 'proof/restored.json').read_text())
    assert restored['restored'] and restored['cache_equal'] and restored['pack_equal']
    assert restored['module'] == proof['baseline']
    baseline_path = ROOT / 'artifacts/update-templates-v1/proof-v2/report.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['complete'] and baseline['baseline_snapshot_restored']
    result = []
    for bank in proof['banks']:
        measured = bank['measurement']
        assert measured['complete']
        prefix = measured['prefix_tokens']
        assert prefix == (38 if bank['bank'] == 'voting' else 27)
        assert all(sha(ROOT / p) == h for p, h in measured['reference_hashes'].items())
        for case in measured['cases']:
            saved = d / 'proof' / bank['bank'] / 'measurement' / f"{case['case']}-r{case['repeat']}"
            paths = sorted(saved.glob('*.json'))
            replies = [json.loads(p.read_text()) for p in paths]
            assert len(replies) == case['update_calls']
            assert sum(r['ok']['progress']['instructions'] for r in replies) == case['total_handler_instructions']
            progress = replies[-1]['ok']['progress']
            assert progress['done']
            ref = ROOT / f"artifacts/boomdao-current-v1/{case['case']}-r1"
            reference = json.loads((ref / 'report.json').read_text())
            for i in range(32):
                p = ref / 'queries' / f'layer-{i:02d}.npy'
                if p.exists():
                    value = np.load(p, allow_pickle=False)
                    if i < 31:
                        value = value[prefix:]
                    assert bits(value) == progress['hidden_hashes'][i]
                else:
                    assert i == 30
                with np.load(ref / 'queries/states' / f'layer-{i:02d}.npz', allow_pickle=False) as z:
                    value = np.concatenate([z['keys'][prefix:].ravel(), z['values'][prefix:].ravel()]) if i % 4 == 3 else z['conv'].ravel()
                assert bits(value) == progress['state_hashes'][i]
            assert bits(progress['final_hidden']) == bits(np.load(ref / 'final-hidden.npy', allow_pickle=False))
            expected = reference['decision_query']['ok']['decision']
            assert {k:v for k,v in expected.items() if k != 'instructions'} == {k:v for k,v in progress['decision'].items() if k != 'instructions'}
            previous = [c for b in baseline['banks'] for c in b['measurement']['cases'] if c['case'] == case['case']]
            assert len(previous) == 3
            assert all(c['tokens'] == case['tokens'] and c['suffix_tokens'] == case['suffix_tokens'] for c in previous)
            before = statistics.median(c['total_handler_instructions'] for c in previous)
            after = case['total_handler_instructions']
            result.append(dict(case=case['case'], baseline_median=before,
                               candidate=after, saved=before-after,
                               reduction_percent=100*(1-after/before),
                               worker_calls=len(replies), target_met=after <= 100_000_000_000,
                               saved_reply_hashes={p.name:sha(p) for p in paths}))
    assert sorted(c['case'] for c in result) == ['617', '620', '653']
    summary = dict(module=proof['candidate'], verified=True, baseline_restored=True,
                   proof_sha256=sha(proof_path), baseline_report_sha256=sha(baseline_path),
                   cases=result, all_targets_met=all(c['target_met'] for c in result),
                   scope='Three current inputs, unchanged voting38/common27 prefixes and weights. One candidate run versus three historical baseline runs. 31 exported hidden layers, 32 conv/KV state hashes, final norm and decision match; layer30 hidden and dense Delta state are not directly exported.')
    (d / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
