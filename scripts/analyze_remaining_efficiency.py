#!/usr/bin/env python3
"""Summarize the complete main-question profile without summing nested spans."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--profile', required=True)
ap.add_argument('--production', required=True)
ap.add_argument('--output', required=True)
a = ap.parse_args()
pp, rp = ROOT / a.profile, ROOT / a.production
profile, production = json.loads(pp.read_bytes()), json.loads(rp.read_bytes())
rows = [r for r in profile['cases'] if r['label'] == 'roll']
assert len(rows) == production['query_count'] == 50
assert sorted(r['index'] for r in rows) == list(range(50))
assert all(r['bitwise_equal'] for r in rows)
assert profile['module_sha256'] == production['wasm_sha256']
spans = Counter()
for r in rows:
    for name, instructions, calls in r['profile']['ok']['spans']:
        assert instructions >= 0 and calls > 0
        spans[name] += instructions
# These measured regions are disjoint at their call sites. Bridge/pair/carry
# and evaluate spans are inclusive parents and must never be added here.
groups = {
    'INT8 base projection and continuation': ['base_project_inclusive', 'integer_k_continue'],
    'F32 LoRA A/B': ['lora_matmul_A', 'lora_matmul_A_down', 'lora_matmul_A_shared',
                    'lora_matmul_B', 'integer_k_finish_B', 'stream_down_A_continue',
                    'stream_input_A_once', 'pair_out_A_continue', 'bridge_q_A_once'],
    'wire decode/encode': ['wire_decode', 'wire_encode'],
    'GQA head evaluation': ['gqa_head_views'],
}
total = sum(r['profile']['ok']['instructions'] for r in rows)
classified = {k: dict(instructions=sum(spans[s] for s in names), spans=names)
              for k, names in groups.items()}
assert sum(x['instructions'] for x in classified.values()) <= total
for x in classified.values():
    x['percent_of_profile'] = 100 * x['instructions'] / total
ordinary = production['total_instructions']
budget = 32 * 5_000_000_000
result = dict(scope='One main BOOM DAO question, 132 tokens, prefix45/suffix87; '
                   '50 replayed profiling queries plus one separate standard terminal diagnostic. '
                   'Profile spans are measured handler regions, not hardware costs; other remains unclassified.',
    module_sha256=profile['module_sha256'], production_instructions=ordinary,
    profile_instructions=total, profile_minus_production=total-ordinary,
    groups=classified, unclassified_instructions=total-sum(x['instructions'] for x in classified.values()),
    nominal_min_queries=math.ceil(ordinary/5_000_000_000),
    target32_nominal_budget=budget, target32_required_reduction=ordinary-budget,
    target32_required_percent=100*(ordinary-budget)/ordinary,
    target32_base_only_required_percent=100*(ordinary-budget)/classified['INT8 base projection and continuation']['instructions'],
    limits='Nominal arithmetic bounds only; IC total accounting, dependency/packet boundaries and 2MB limits make packing stricter.',
    profile_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),
    production_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out = ROOT / a.output
if out.exists():
    raise ValueError('Do not replace existing evidence')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('groups', 'scope', 'limits')}))
