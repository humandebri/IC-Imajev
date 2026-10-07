#!/usr/bin/env python3
"""Verify archived S1 replies; do not treat inline counters as cost attribution."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/s1-inner-profile-v1'
B = D / 'build-v6'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    build = json.loads((B / 'report.json').read_text())
    report = json.loads((D / 'check/report.json').read_text())
    for manifest in [build['source_hashes'], build['dependency_hashes'],
                     build['helper_dependency_hashes'], report['source_hashes']]:
        assert all(sha(ROOT / p) == h for p, h in manifest.items())
    assert sha(B / 'diagnostic.wasm') == build['wasm_sha256'] == report['wasm_sha256']
    oracle = ROOT / 'artifacts/bounded_i16/native/release/wat_s1_args'
    cases = []
    assert len(report['cases']) == 19 and report['ordinary_queries'] == 38
    for case in report['cases']:
        assert sha(D / 'check' / (case['label'] + '.input.bin')) == case['input_sha256']
        for name, measured in case['measurements'].items():
            reply = ROOT / measured['reply']
            assert sha(reply) == measured['reply_sha256']
            decoded = json.loads(subprocess.check_output(
                [str(oracle), 'decode', str(reply), 'measurement'], text=True))
            assert all(measured[k] == v for k, v in decoded.items())
            assert decoded['digest'] == case['native']['digest']
            assert decoded['total_instructions'] == sum(decoded[k] for k in
                ['quantize_instructions', 'input_prepare_instructions', 'project_instructions'])
            if name == 'instrumented_s1':
                inner = json.loads(subprocess.check_output(
                    [str(B / 'tool'), 'decode', str(reply)], text=True))
                assert inner == measured['inner_profile'] and len(inner) == 4
                assert inner[0] > 0 and inner[3] > 0
                assert (inner[1] > 0) == (case['tokens'] >= 4)
                assert (inner[2] > 0) == (case['tokens'] >= 3 and case['tokens'] % 2 == 1)
                assert sum(inner) <= decoded['project_instructions']
        control = case['measurements']['s1_pair_bounds']['total_instructions']
        instrumented = case['measurements']['instrumented_s1']['total_instructions']
        cases.append(dict(label=case['label'], tokens=case['tokens'],
                          control=control, instrumented=instrumented,
                          overhead_percent=100 * (instrumented / control - 1),
                          raw_inner=case['measurements']['instrumented_s1']['inner_profile']))
    summary = dict(module=build['wasm_sha256'], ordinary_queries=38,
                   native_bits_equal=True, saved_replies_verified=True,
                   adopted=False, cost_attribution_validated=False,
                   limitation='Inline counter boundaries can move ahead-of-time block charges between spans. Raw counters are diagnostic observations, not disjoint operation costs. Full production inference was not changed.',
                   report_sha256=sha(D / 'check/report.json'),
                   helper_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                                  [oracle, B / 'tool', B / 'imports-tool']}, cases=cases)
    (D / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    files = [Path(__file__), ROOT / 'scripts/build_s1_inner_profile.py',
             ROOT / 'scripts/check_s1_inner_profile.py', ROOT / 'scripts/s1_inner_profile_tool.rs',
             D / 'frozen-check.py', D / 'check-upstream.sha256', B / 'report.json',
             B / 'imports.json', B / 'kernel.wat', D / 'summary.json'] + list((B / 'src').glob('*.rs'))
    with zipfile.ZipFile(D / 'frozen-workflow.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(ROOT)))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
