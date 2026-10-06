#!/usr/bin/env python3
"""Verify archived MLP complete diagnostics; no whole-graph query reduction implied."""
import hashlib
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import decode

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
build = ROOT / 'artifacts/prefix_codec/full-build-mlp-complete-v1'
module = sha(build / 'full.wasm')
build_sources = json.loads((build / 'source-hashes.json').read_text())
rows = []
reports = {}
for label in ['617', 'insufficient', 'maximum']:
    d = ROOT / f'artifacts/f32_k_continue/complete-check-{label}'
    r = json.loads((d / 'report.json').read_text())
    assert r['wasm_sha256'] == module and r['goal_50_verified'] is False
    assert len(r['cases']) == 15 and r['regular_diagnostic_queries'] == 84
    assert r['profile_diagnostic_queries'] == 6
    with zipfile.ZipFile(d / 'validated-source.zip') as archive:
        for name, expected in r['source_hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
            assert sha(ROOT / name) == expected, name
            if name in build_sources:
                assert build_sources[name] == expected, name
    source = ROOT / f'artifacts/prefix_codec/full-capture-prepared-proof/{label}'
    frozen = json.loads((source / 'report.json').read_text())
    for row in r['cases']:
        original = next(q for q in frozen['queries'] if q['op'] == 'mlp_full_integer'
                        and f".layers.{row['layer']}." in q['tensor'])
        request = source / 'queries' / f"{original['index']:06d}.request.bin"
        response = source / 'queries' / f"{original['index']:06d}.response.bin"
        assert sha(request) == row['source_request_sha256']
        assert sha(response) == row['source_response_sha256']
        _, expected = decode(response.read_bytes())
        assert row['bitwise_hidden_norm_equal'] and row['fused_complete']
        assert row['bitwise_prepared_equal'] is None
        for calls in [row['calls'], row['same_module_split_baseline']['calls']]:
            _, actual = decode((d / f"{calls[-1]['index']:06d}.response.bin").read_bytes())
            assert actual.tobytes() == expected.tobytes()
        assert row['ordinary_queries'] == len(row['chunks'])
        assert row['same_module_split_baseline']['ordinary_queries'] == len(row['chunks']) + 1
        assert row['total_instructions'] == sum(c['ok']['instructions'] for c in row['calls'])
        assert row['candid_bytes'] == sum(c['ok']['request_bytes'] + c['ok']['reply_bytes'] for c in row['calls'])
        assert row['max_query_instructions'] <= 5_000_000_000
        assert row['instruction_delta'] == row['total_instructions'] - row['same_module_split_baseline']['total_instructions'] < 0
        assert row['candid_bytes_delta'] == row['candid_bytes'] - row['same_module_split_baseline']['candid_bytes'] < 0
        rows.append(row)
    reports[label] = sha(d / 'report.json')
full_dir = ROOT / 'artifacts/prefix_codec/full-mlp-complete-proof'
full = json.loads((full_dir / 'report.json').read_text())
assert full['wasm_sha256'] == module and full['goal_50_verified'] is False
assert len(full['cases']) == 6 and all(c['full_bitwise_equal'] for c in full['cases'])
with zipfile.ZipFile(full_dir / 'validated-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, expected in full['source_hashes'].items():
        assert sha(ROOT / name) == expected, name
        archive.write(ROOT / name, name)
for case in full['cases']:
    report = full_dir / case['label'] / 'report.json'
    assert sha(report) == case['report_sha256']
    assert case['max_query_instructions'] <= 5_000_000_000
    original = ROOT / f"artifacts/output-pairs-v2-{case['baseline']}/report.json"
    assert sha(original) == case['baseline_report_sha256']
    actual = json.loads(report.read_text())
    assert actual['replayed_queries'] == 0
    failures = report.parent / 'queries/failures.jsonl'
    assert not failures.exists() or not failures.read_bytes()
main = next(c for c in full['cases'] if c['label'] == '617')
assert main['query_count'] == 62
result = dict(scope=__doc__, wasm_sha256=module, report_hashes=reports,
              cases=len(rows), hidden_norm_bitwise_equal=True,
              instruction_delta_range=[min(c['instruction_delta'] for c in rows), max(c['instruction_delta'] for c in rows)],
              candid_bytes_delta_range=[min(c['candid_bytes_delta'] for c in rows), max(c['candid_bytes_delta'] for c in rows)],
              ordinary_diagnostic_queries=252, profile_diagnostic_queries=18,
              full_regression_report_sha256=sha(full_dir / 'report.json'),
              actual_whole_graph_queries=main['query_count'], goal_50_verified=False)
(ROOT / 'artifacts/f32_k_continue/mlp-complete-archive-check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
