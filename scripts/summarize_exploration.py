#!/usr/bin/env python3
"""Keep sampled cost estimates, exact payload counts and accuracy screens distinct."""
import collections
import hashlib
import json
import pathlib
import statistics
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    full = read('artifacts/layout-full-617/first-report.json')
    counts = collections.Counter()
    for q in full['queries']:
        if 'integer' not in q['op']:
            continue
        path = ROOT / f"artifacts/layout-full-617/queries/{q['index']:06d}.request.bin"
        with path.open('rb') as f:
            size, = struct.unpack('<I', f.read(4))
            h = json.loads(f.read(size))
        counts[q['op'], tuple(h['dims'][:3])] += 1
    reports = {name: read(f'artifacts/bottleneck/{name}/report.json') for name in ('explore-load-baseline', 'explore-load-combined', 'explore-current-profile')}
    weighted = {}
    spans = collections.Counter()
    profile_overhead = 0
    for name, report in reports.items():
        assert len(report['cases']) == len(counts)
        weighted[name] = 0
        for case in report['cases']:
            assert case['bitwise_equal']
            frequency = counts[case['op'], tuple(case['dims'][:3])]
            weighted[name] += frequency * case['normal']['instructions']
            if name == 'explore-current-profile':
                profile_overhead += frequency * case['profile_overhead']
                for span, cost, _ in case['profile']['spans']:
                    spans[span] += frequency * cost
    rotations = {}
    for label, path in [('617', 'docs/rotation-screen.json'), ('insufficient', 'docs/rotation-insufficient-screen.json'), ('maximum', 'docs/rotation-maximum-screen.json')]:
        data = read(path)
        rotations[label] = {}
        for key in ('rmse_ratio', 'weight_only_rmse_ratio', 'activation_per_row_rmse_ratio'):
            values = [v[key] for c in data['cases'] for v in c['rotations']]
            rotations[label][key] = dict(samples=len(values), improved=sum(v < 1 for v in values), median=statistics.median(values), minimum=min(values), maximum=max(values))
    baseline_hash = hashlib.sha256((ROOT / 'artifacts/exploration/fused-load/baseline.wasm').read_bytes()).hexdigest()
    candidate_hash = hashlib.sha256((ROOT / 'artifacts/exploration/fused-load/candidate.wasm').read_bytes()).hexdigest()
    out = dict(scope='Representative-query span expansion, not a new full-model run; inclusive span parents must not be added to children; rotation errors are base-projection samples, not accuracy', baseline_wasm_sha256=baseline_hash, load_candidate_wasm_sha256=candidate_hash, load_candidate_binary_identical=baseline_hash == candidate_hash, covered_shapes=len(counts), covered_projection_queries=sum(counts.values()), weighted_sample_instructions=weighted, profiled_call_extra_instructions=profile_overhead, profiling_build_normal_cost_change=weighted['explore-current-profile']/weighted['explore-load-baseline']-1, spans={name: dict(estimated_instructions=cost, fraction_of_baseline_full=cost/full['total_instructions']) for name,cost in spans.items()}, rotation_screens=rotations, payloads={label: read(path)['totals'] for label,path in [('prefix','docs/query-payload-exploration.json'),('standalone','docs/query-payload-standalone-exploration.json')]})
    (ROOT / 'docs/exploration-results.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rotation_screens','spans','payloads')}, indent=2))


if __name__ == '__main__':
    main()
