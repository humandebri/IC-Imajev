#!/usr/bin/env python3
"""Verify raw prepared-S3 replies and reject the measured instruction regression."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/s3-prepared-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r = json.loads((D / 'check/report.json').read_text())
    b = json.loads((D / 'build/report.json').read_text())
    helper = ROOT / 'artifacts/bounded_i16/native/release/wat_s1_args'
    assert sha(D / 'build/diagnostic.wasm') == r['wasm_sha256'] == b['wasm_sha256']
    for manifest in [r['source_hashes'], b['source_hashes'], b['dependency_hashes']]:
        assert all(sha(ROOT / p) == h for p, h in manifest.items())
    assert len(r['cases']) == 19 and r['ordinary_queries'] == 38
    cases = []
    for c in r['cases']:
        assert c['bitwise_equal']
        assert sha(D / 'check' / (c['label'] + '.input.bin')) == c['input_sha256']
        for name in ['s1_pair_bounds', 's3_prepared']:
            m = c['measurements'][name]
            reply = ROOT / m['reply']
            assert sha(reply) == m['reply_sha256']
            raw = json.loads(subprocess.check_output([str(helper), 'decode', str(reply), 'measurement'], text=True))
            assert all(m[field] == value for field, value in raw.items())
            assert raw['digest'] == c['native']['digest']
            assert raw['total_instructions'] == sum(raw[k] for k in ['quantize_instructions', 'input_prepare_instructions', 'project_instructions'])
        before = c['measurements']['s1_pair_bounds']['total_instructions']
        after = c['measurements']['s3_prepared']['total_instructions']
        assert abs(c['reduction_percent'] - 100 * (1 - after / before)) < 1e-10
        cases.append(dict(label=c['label'], tokens=c['tokens'], before=before, after=after,
                          increase_percent=100*(after/before-1)))
    seal = r['preparations'][-1]
    reply = ROOT / seal['reply']
    assert sha(reply) == seal['reply_sha256']
    raw = json.loads(subprocess.check_output([str(helper), 'decode', str(reply), 'preparation'], text=True))
    assert all(seal[k] == v for k, v in raw.items())
    summary = dict(module=r['wasm_sha256'], saved_replies_verified=True, native_bits_equal=True,
                   ordinary_queries=38, adopted=False, prepared_weight_bytes_ratio=b['weight_bytes_ratio'],
                   seal=raw, cases=cases,
                   scope='Projection only; immutable preparation excluded equally from inference measurements. Expanded transformed weights did not overcome query operand/reconstruction/loading costs.')
    (D / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    paths = [Path(__file__), ROOT / 'scripts/build_s3_prepared_probe.py', ROOT / 'scripts/check_s3_prepared_probe.py',
             ROOT / 'scripts/check_s3_stream_probe.py', D / 'frozen-check.py', D / 'check-upstream.sha256',
             D / 'build/report.json', D / 'build/kernel.wat', D / 'summary.json'] + list((D / 'build/src').glob('*.rs'))
    with zipfile.ZipFile(D / 'frozen-workflow.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, str(p.relative_to(ROOT)))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
