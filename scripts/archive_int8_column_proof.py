#!/usr/bin/env python3
"""Freeze exact INT8 continuation evidence without claiming graph speedup."""
import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


build = ROOT / 'artifacts/prefix_codec/full-build-int8-column-v2'
directory = ROOT / 'artifacts/prefix_codec/full-int8-column-proof'
full = json.loads((directory / 'report.json').read_bytes())
before = json.loads((ROOT / 'artifacts/prefix_codec/full-mlp-stream-proof/report.json').read_bytes())
hashes = json.loads((build / 'source-hashes.json').read_bytes())
assert all(sha(ROOT / path) == value for path, value in hashes.items())
assert full['wasm_sha256'] == sha(build / 'full.wasm')
assert all(sha(ROOT / path) == value for path, value in full['source_hashes'].items())
path = ROOT / 'artifacts/f32_k_continue/int8-column-check/report.json'
diagnostic = json.loads(path.read_bytes())
assert diagnostic['wasm_sha256'] == full['wasm_sha256']
assert all(sha(ROOT / p) == v for p, v in diagnostic['source_hashes'].items())
assert len(diagnostic['cases']) == 32
assert {(c['label'], c['layer'], c['tokens']) for c in diagnostic['cases']} == {
    (label, layer, n) for label, n in [('prefix', 45), ('617', 87), ('insufficient', 80), ('maximum', 89)]
    for layer in [0, 30]
}
assert all(c['base_bitwise_equal'] and c['a_bitwise_equal'] and c['lora_bitwise_equal']
           and c['maximum_query_instructions'] <= 5_000_000_000 for c in diagnostic['cases'])
assert diagnostic['regular_diagnostic_queries'] == 216
assert diagnostic['profile_diagnostic_queries'] == 3
assert diagnostic['rejected_ordinary_queries'] == len(diagnostic['rejected']) == 6
spans = [{name: (cost, count) for name, cost, count in p['ok']['spans']}
         for p in diagnostic['profiles']]
assert all(s['integer_k_quantize_new_columns'][1] == 1 and s['integer_k_continue'][1] == 1
           for s in spans[:2])
assert spans[2]['integer_k_finish_B'][1] == 1
assert not any('quantize' in name or 'continue' in name or name.endswith('_A') for name in spans[2])
changes = []
for case in full['cases']:
    old = next(c for c in before['cases'] if c['label'] == case['label'])
    assert case['full_bitwise_equal']
    assert case['query_count'] == old['query_count']
    assert case['total_candid_bytes'] == old['total_candid_bytes']
    changes.append(dict(label=case['label'], query_count=case['query_count'],
                        before_instructions=old['total_instructions'],
                        after_instructions=case['total_instructions'],
                        instruction_delta=case['total_instructions'] - old['total_instructions'],
                        candid_bytes=case['total_candid_bytes'],
                        max_query_instructions=case['max_query_instructions'],
                        heap_bytes=case['max_observed_heap_bytes'], wall_seconds=case['wall_seconds']))
bridge = json.loads((ROOT / 'artifacts/prefix_codec/full-build-host-checksum-v3/bridge.json').read_bytes())
assert all(sha(ROOT / p) == v for p, v in bridge.items())
(build / 'bridge.json').write_text(json.dumps(bridge, indent=2) + '\n')
with zipfile.ZipFile(directory / 'validated-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in dict.fromkeys([*full['source_hashes'], *hashes,
                              'scripts/check_int8_column_continue.py', 'scripts/archive_int8_column_proof.py']):
        archive.write(ROOT / name, name)
(directory / 'before-after.json').write_text(json.dumps(dict(
    cases=changes, diagnostic_report_sha256=sha(path),
    scope='Exact INT8/F32 column continuation component verified; unchanged inference graph still needs fusion.',
    goal_50_verified=False), indent=2) + '\n')
print(json.dumps(changes, indent=2))
