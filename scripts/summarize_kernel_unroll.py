#!/usr/bin/env python3
"""Summarize fresh full-graph kernel trials with source-bound comparisons."""
import argparse, collections, hashlib, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='kernel-unroll-v1')
    ap.add_argument('--baseline', default='terminal-v3')
    args = ap.parse_args()
    summary = {}
    for name in ['prefix', '617', 'insufficient', 'maximum', 'normal']:
        directory = ROOT / f'artifacts/{args.run_name}-{name}'
        raw = (directory / 'report.json').read_bytes()
        r = json.loads(raw)
        before = json.loads((ROOT / f'artifacts/{args.baseline}-{name}/report.json').read_text())
        c = json.loads((ROOT / f'docs/{args.run_name}-{name}-results.json').read_text())
        assert c['all_retained_hidden_bitwise_equal']
        assert c['retained_state_arrays_bitwise_equal'] == (72 if name == 'prefix' else 48)
        assert c['decision_identical'] is (None if name == 'prefix' else True)
        assert r['replayed_queries'] == 0
        verification = ('before-after' if 'deployed_wasm_sha256' in r else 'post-run')
        module_hash = (r['deployed_wasm_sha256'] if verification == 'before-after' else json.loads((directory / 'post-run-module-verification.json').read_text())['wasm_sha256'])
        assert r['wasm_sha256'] == module_hash
        failures = directory / 'queries/failures.jsonl'
        assert not failures.exists() or not failures.read_text().strip()
        metrics = ['query_count', 'total_instructions', 'total_candid_bytes', 'wall_seconds_this_run', 'max_query_instructions', 'max_observed_heap_bytes']
        ops = collections.defaultdict(lambda: dict(queries=0, instructions=0))
        for q in r['queries']:
            ops[q['op']]['queries'] += 1
            ops[q['op']]['instructions'] += q['ok']['instructions']
        summary[name] = dict(before={k: before[k] for k in metrics}, after={k:r[k] for k in metrics}, instruction_reduction=1-r['total_instructions']/before['total_instructions'], tokens=r['tokens'], wasm_sha256=r['wasm_sha256'], module_verification=verification, ops=dict(ops), comparison=r.get('comparison'), raw_report_sha256=hashlib.sha256(raw).hexdigest(), zero_failures=True, zero_replay=True)
        print(name, json.dumps({k:v for k,v in summary[name].items() if k != 'ops'}))
    (ROOT / f'docs/{args.run_name}-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
if __name__ == '__main__':
    main()
