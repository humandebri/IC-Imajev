#!/usr/bin/env python3
"""Re-decode both current-register and stack Delta output/final-state evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/delta-stack-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    build = json.loads((D / 'build/report.json').read_text())
    report = json.loads((D / 'check/report.json').read_text())
    assert sha(D / 'build/diagnostic.wasm') == build['wasm_sha256'] == report['wasm_sha256']
    for manifest in [build['source_hashes'], build['dependency_hashes'], report['source_hashes']]:
        assert all(sha(ROOT / p) == h for p,h in manifest.items())
    assert len(report['cases']) == 7 and report['ordinary_queries'] == 14
    helper = ROOT / 'artifacts/delta-writeback-target/debug/writeback_args'
    cases = []
    for case in report['cases']:
        assert case['bitwise_equal']
        for name in ['register', 'stack']:
            m = case['measurements'][name]
            reply = ROOT / m['reply']
            assert sha(reply) == m['reply_sha256']
            raw = json.loads(subprocess.check_output([str(helper), 'decode', str(reply)], text=True))
            assert all(m[k] == v for k,v in raw.items())
            assert bytes(raw['digest']).hex() == case['native_digest']
            assert raw['kernel_instructions'] <= raw['handler_instructions']
        before = case['measurements']['register']['kernel_instructions']
        after = case['measurements']['stack']['kernel_instructions']
        assert abs(case['reduction_percent'] - 100*(1-after/before)) < 1e-10
        cases.append(dict(tokens=case['tokens'], before=before, after=after,
                          reduction_percent=case['reduction_percent']))
    summary = dict(module=report['wasm_sha256'], native_output_and_final_state_bits_equal=True,
                   saved_replies_verified=True, ordinary_queries=14, cases=cases,
                   scope='Synthetic single 128x128 head, same-module current register control. Full inference still requires verification.')
    (D / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    paths = [Path(__file__), ROOT / 'scripts/build_delta_stack_probe.py', ROOT / 'scripts/check_delta_stack_probe.py',
             ROOT / 'scripts/check_delta_register_probe.py', D / 'frozen-check.py', D / 'check-upstream.sha256',
             D / 'build/report.json', D / 'build/lib.rs', D / 'build/delta_simd.rs', D / 'build/kernel.wat', D / 'summary.json']
    with zipfile.ZipFile(D / 'frozen-workflow.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, str(p.relative_to(ROOT)))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
