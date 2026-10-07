#!/usr/bin/env python3
"""Paired local IC full-forward measurement with one archived Wasm for both packs.

Weight upload/preparation must finish first. Every measured forward uses a fresh
journal, so recorded replies are never replayed. Query time includes the local
bridge/serialization/HTTP; counters are the existing handler performance counter.
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport
from prefix_inference import verify_module

VARIANTS = {
    'before': ('2hlkr-gl777-77775-aaaxa-cai', 'checkpoints/full-int8.manifest.json'),
    'after': ('2akmf-lt777-77775-aaaxq-cai', 'artifacts/merged-adapter-v2/model.manifest.json'),
}
WASM = 'artifacts/prepared-f32/candidate.wasm'
WASM_HASH = '86d93cda02bb5bb6c6e4d17bea38b3315206aace76cdc94d3a4941a6edbec0df'
FIXTURE = 'artifacts/merged-adapter-v2/before.json'
DIRECTORY = ROOT / 'artifacts/merged-adapter-ic-v1'
FLAGS = ['--arithmetic', 'int8', '--wire-codec', 'bf16-exact', '--compact-lossless',
         '--fuse-add-norm', '--fuse-norm-rope', '--fuse-delta', '--compact-heads',
         '--delta-head-cap', '8', '--attention-head-cap', '4', '--row-cap', '3072',
         '--work-cap', '3000000000']


def cache_status(variant):
    canister, manifest_path = VARIANTS[variant]
    manifest = json.loads((ROOT / manifest_path).read_text())
    t = Transport(manifest['model'], 'http://localhost:8001/', canister,
                  str(ROOT / 'artifacts/imajev-local.pem'), DIRECTORY / f'{variant}-status', manifest['pack_hash'])
    try:
        verify_module(t, WASM_HASH)
        pack = t.command(dict(op='pack_status'))['ok']
        if not pack['ready'] or pack['pack_hash'] != manifest['pack_hash']:raise ValueError('Prepared pack identity')
        status = t.command(dict(op='weight_cache_status'))['ok']['cache']
        expected = [w for w in manifest['tensors'] if w['dtype'] in ('int8', 'f32') and 'embed_tokens' not in w['name']]
        if status['bytes'] != sum(w['bytes'] for w in expected) or set(status['names']) != {w['name'] for w in expected}:raise ValueError('Prepared weight cache coverage')
        return status
    finally:
        t.close()


def summarize(repeats):
    results = []
    for repeat in range(repeats):
        pair = {}
        for variant in VARIANTS:
            run = DIRECTORY / f'{variant}-{repeat}'
            r = json.loads((run / 'report.json').read_text())
            decision = r['decision_query']['ok']['decision']
            if r['replayed_queries'] != 0 or r['executed_query_count'] != r['query_count'] or (run / 'queries/failures.jsonl').exists():raise ValueError('Replay or failed-query run cannot be counted as clean measurement')
            if r['wasm_sha256'] != WASM_HASH or r['deployed_wasm_sha256'] != WASM_HASH or len(r['layers']) != 32 or r['tokens'] != 132:raise ValueError('Full graph identity')
            canister, manifest_path = VARIANTS[variant]
            manifest = json.loads((ROOT / manifest_path).read_text())
            if r['canister'] != canister or r['pack_hash'] != manifest['pack_hash']:raise ValueError('Run pack identity')
            if r.get('request_step_offset') != (repeat+1)*100000:raise ValueError('Measurement request nonce')
            pair[variant] = dict(value=decision['value'], abstained=decision['abstained'], probabilities=decision['probabilities'],
                                 unknown_probability=decision['unknown_probability'], raw_logits=decision['raw_logits'],
                                 input_hash=r['input_hash'], pack_hash=r['pack_hash'], instructions=r['total_instructions'],
                                 query_count=r['query_count'], query_wall_seconds=sum(q['wall_seconds'] for q in r['queries']),
                                 runner_wall_seconds=r['wall_seconds_this_run'], candid_bytes=r['total_candid_bytes'],
                                 max_query_instructions=r['max_query_instructions'], observed_heap_bytes=r['max_observed_heap_bytes'])
            pair[variant]['layer_instructions'] = [layer['instructions'] for layer in r['layers']]
        if pair['before']['input_hash'] != pair['after']['input_hash']:raise ValueError('Input mismatch')
        pair['repeat'] = repeat
        pair['decision_equal'] = (pair['before']['value'], pair['before']['abstained']) == (pair['after']['value'], pair['after']['abstained'])
        pair['probability_max_abs_difference'] = max(abs(x-y) for x,y in zip(pair['before']['probabilities']+[pair['before']['unknown_probability']],pair['after']['probabilities']+[pair['after']['unknown_probability']]))
        results.append(pair)
    for variant in VARIANTS:
        if len({(r[variant]['instructions'], json.dumps(r[variant]['raw_logits'])) for r in results}) != 1:raise ValueError('Non-deterministic repeated instructions or logits')
    b, a = [results[0][name]['instructions'] for name in ('before','after')]
    before_times = [r['before']['runner_wall_seconds'] for r in results]
    after_times = [r['after']['runner_wall_seconds'] for r in results]
    report = dict(scope='Local IC ordinary-query full 32-layer prefill, identical 132-token input; one archived Wasm and common unfused-LoRA projection schedule. Not production network latency or the latest fused query32 route.',
                  api_url='http://localhost:8001/', wasm_sha256=WASM_HASH, flags=FLAGS, repeats=repeats, runs=results,
                  instruction_scope='Existing handler counters including binary frame work, excluding CDK Candid decode/encode; includes decision handler.',
                  timing_scope='Full runner wall clock, module verification/bridge/serialization/HTTP/file journals included; upload/preparation excluded.',
                  query_cache_mitigation='Each repeat starts at a unique step offset; encoded query arguments differ, and journals are fresh. Node cache configuration is unchanged.',
                  instruction_reduction=1-a/b, instructions_before=b, instructions_after=a,
                  median_seconds_before=statistics.median(before_times), median_seconds_after=statistics.median(after_times),
                  observed_median_speedup=statistics.median(before_times)/statistics.median(after_times),
                  seconds_before=before_times, seconds_after=after_times,
                  time_reduction=1-statistics.median(after_times)/statistics.median(before_times),
                  per_layer=[dict(layer=i,before=results[0]['before']['layer_instructions'][i],after=results[0]['after']['layer_instructions'][i]) for i in range(32)],
                  all_decisions_equal=all(r['decision_equal'] for r in results),
                  probability_max_abs_difference=max(r['probability_max_abs_difference'] for r in results))
    (DIRECTORY / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','flags')}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.repeats <= 10:raise ValueError('Repeat count')
    if hashlib.sha256((ROOT / WASM).read_bytes()).hexdigest() != WASM_HASH:raise ValueError('Archived Wasm hash')
    if args.summary_only:
        summarize(args.repeats)
        return
    snapshot = {v: cache_status(v) for v in VARIANTS}
    sources = ['scripts/run_full_canister.py','scripts/measure_merged_adapter_ic.py','client/full_inference.py','client/transport.py','client/prefix_inference.py']
    source_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    metadata = dict(canisters={v: data[0] for v,data in VARIANTS.items()}, wasm=WASM, wasm_sha256=WASM_HASH,
                    fixture=FIXTURE, flags=FLAGS, cache_before=snapshot, script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                    client_source_hashes=source_hashes, archived_build_source_hashes=json.loads((ROOT/'artifacts/prepared-f32/build-source-hashes.json').read_text()))
    (DIRECTORY / 'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for repeat in range(args.repeats):
        for variant in (['before','after'] if repeat%2==0 else ['after','before']):
            run = DIRECTORY / f'{variant}-{repeat}'
            if run.exists():raise ValueError(f'Refusing to replay an existing run: {run}')
            canister, manifest = VARIANTS[variant]
            command = [sys.executable, str(ROOT / 'scripts/run_full_canister.py'), '--canister',canister,
                       '--wasm',WASM,'--manifest',manifest,'--reference',FIXTURE,'--record','0','--directory',str(run),'--step-offset',str((repeat+1)*100000),*FLAGS]
            start = time.monotonic()
            print(json.dumps(dict(variant=variant,repeat=repeat,event='start')),flush=True)
            with (DIRECTORY / f'{variant}-{repeat}.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=log)
            print(json.dumps(dict(variant=variant,repeat=repeat,event='complete',process_wall_seconds=time.monotonic()-start)),flush=True)
    metadata['cache_after'] = {v: cache_status(v) for v in VARIANTS}
    if any(metadata['cache_after'][v]['bytes'] != snapshot[v]['bytes'] or set(metadata['cache_after'][v]['names']) != set(snapshot[v]['names']) for v in VARIANTS):raise ValueError('Weight cache changed during measurement')
    if source_hashes != {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}:raise ValueError('Client source changed during measurement')
    (DIRECTORY / 'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    summarize(args.repeats)


if __name__ == '__main__':
    main()
