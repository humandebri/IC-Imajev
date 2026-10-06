#!/usr/bin/env python3
"""Compare selected-column operand preparation against frozen full-stride work."""
import hashlib
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


build = ROOT / 'artifacts/prefix_codec/full-build-int8-range-v2'
directory = ROOT / 'artifacts/prefix_codec/full-int8-range-v2-proof'
diagnostic_directory = ROOT / 'artifacts/f32_k_continue/int8-range-v2-check'
previous_directory = ROOT / 'artifacts/f32_k_continue/int8-column-check'
full = json.loads((directory / 'report.json').read_bytes())
old_full = json.loads((ROOT / 'artifacts/prefix_codec/full-int8-column-proof/report.json').read_bytes())
diagnostic = json.loads((diagnostic_directory / 'report.json').read_bytes())
previous = json.loads((previous_directory / 'report.json').read_bytes())
hashes = json.loads((build / 'source-hashes.json').read_bytes())
assert all(sha(ROOT / p) == v for p, v in hashes.items())
assert full['wasm_sha256'] == diagnostic['wasm_sha256'] == sha(build / 'full.wasm')
assert all(sha(ROOT / p) == v for p, v in diagnostic['source_hashes'].items())
assert all(sha(ROOT / p) == v for p, v in full['source_hashes'].items())
assert diagnostic['regular_diagnostic_queries'] == 216
assert diagnostic['profile_diagnostic_queries'] == 3
assert diagnostic['rejected_ordinary_queries'] == len(diagnostic['rejected']) == 6
for index in range(216):
    path = f'{index:06d}.response.bin'
    assert decode((diagnostic_directory / path).read_bytes())[1].tobytes() == decode((previous_directory / path).read_bytes())[1].tobytes()
changes = []
for case in diagnostic['cases']:
    old = next(c for c in previous['cases'] if (c['label'], c['layer'], c['chunks']) == (case['label'], case['layer'], case['chunks']))
    assert case['base_bitwise_equal'] and case['a_bitwise_equal'] and case['lora_bitwise_equal']
    assert case['ordinary_queries'] == old['ordinary_queries']
    assert case['candid_bytes'] == old['candid_bytes']
    changes.append(dict(label=case['label'], layer=case['layer'], chunks=case['chunks'],
                        before_instructions=old['instructions'], after_instructions=case['instructions'],
                        instruction_delta=case['instructions'] - old['instructions']))
assert len(changes) == 32
spans = [{name: (cost, count) for name, cost, count in p['ok']['spans']} for p in diagnostic['profiles']]
assert all(s['integer_k_prepare_new_columns'][1] == 1 for s in spans[:2])
assert 'integer_k_prepare_new_columns' not in spans[2]
graph = []
for case in full['cases']:
    old = next(c for c in old_full['cases'] if c['label'] == case['label'])
    assert case['full_bitwise_equal']
    assert case['query_count'] == old['query_count'] and case['total_candid_bytes'] == old['total_candid_bytes']
    graph.append(dict(label=case['label'], query_count=case['query_count'],
                      total_instructions=case['total_instructions'],
                      instruction_delta=case['total_instructions'] - old['total_instructions'],
                      candid_bytes=case['total_candid_bytes'],
                      max_query_instructions=case['max_query_instructions'],
                      heap_bytes=case['max_observed_heap_bytes'], wall_seconds=case['wall_seconds']))
bridge = json.loads((ROOT / 'artifacts/prefix_codec/full-build-host-checksum-v3/bridge.json').read_bytes())
assert all(sha(ROOT / p) == v for p, v in bridge.items())
(build / 'bridge.json').write_text(json.dumps(bridge, indent=2) + '\n')
head_path = ROOT / 'artifacts/f32_k_continue/delta-head-reuse/report.json'
head = json.loads(head_path.read_bytes())
assert head['wasm_sha256'] == full['wasm_sha256']
assert all(sha(ROOT / p) == v for p, v in head['source_hashes'].items())
assert len(head['cases']) == 12 and head['regular_diagnostic_queries'] == 36
assert head['profile_diagnostic_queries'] == 3
assert all(c['gated_bitwise_equal'] and c['history_bitwise_equal'] for c in head['cases'])
head_spans = [{name: (cost, count) for name, cost, count in p['ok']['spans']} for p in head['profiles']]
assert head_spans[0]['activation_quantize'][1] == 1 and head_spans[0]['lora_matmul_A'][1] == 2
assert all('activation_quantize' not in s and 'lora_matmul_A' not in s for s in head_spans[1:])
with zipfile.ZipFile(directory / 'validated-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in dict.fromkeys([*full['source_hashes'], *hashes,
                           'scripts/check_int8_column_continue.py', 'scripts/archive_int8_range_proof.py',
                           'scripts/check_zero_product_blocks.py', 'scripts/check_delta_head_reuse.py']):
        archive.write(ROOT / p, p)
result = dict(diagnostics=changes, graph=graph, head_reuse_report_sha256=sha(head_path), goal_50_verified=False,
              scope='Repeated inactive-column coefficient transformation removed; graph not yet connected to new continuation.')
(directory / 'before-after.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
