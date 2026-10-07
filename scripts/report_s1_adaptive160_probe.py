#!/usr/bin/env python3
"""Re-decode adaptive S1 evidence and compare it with the rejected padded160 probe."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/s1-adaptive160-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    b = json.loads((D / 'build/report.json').read_text())
    r = json.loads((D / 'check/report.json').read_text())
    assert sha(D / 'build/diagnostic.wasm') == b['wasm_sha256'] == r['wasm_sha256']
    for manifest in [b['source_hashes'], b['dependency_hashes'], r['source_hashes']]:
        assert all(sha(ROOT / p) == h for p,h in manifest.items())
    assert all(p['wasmparser_validation'] for p in b['patches'])
    assert len({p['function_index'] for p in b['patches']}) == 3
    helper = ROOT / 'artifacts/bounded_i16/native/release/wat_s1_args'
    assert len(r['cases']) == 19 and r['ordinary_queries'] == 38
    old = json.loads((ROOT / 'artifacts/s1-output160-v1/check/report.json').read_text())
    previous = {c['label']:c for c in old['cases']}
    cases = []
    for c in r['cases']:
        assert c['bitwise_equal']
        assert sha(D / 'check' / (c['label'] + '.input.bin')) == c['input_sha256']
        assert c['input_sha256'] == previous[c['label']]['input_sha256']
        for name in ['s1_pair_bounds', 'adaptive160']:
            m = c['measurements'][name]
            reply = ROOT / m['reply']
            assert sha(reply) == m['reply_sha256']
            raw = json.loads(subprocess.check_output([str(helper), 'decode', str(reply), 'measurement'], text=True))
            assert all(m[k] == v for k,v in raw.items())
            assert raw['digest'] == c['native']['digest']
            assert raw['total_instructions'] == sum(raw[k] for k in ['quantize_instructions', 'input_prepare_instructions', 'project_instructions'])
        before = c['measurements']['s1_pair_bounds']['total_instructions']
        after = c['measurements']['adaptive160']['total_instructions']
        assert abs(c['reduction_percent']-100*(1-after/before)) < 1e-10
        cases.append(dict(label=c['label'], tokens=c['tokens'], before=before, after=after,
                          reduction_percent=c['reduction_percent'],
                          padded160=previous[c['label']]['measurements']['output160']['total_instructions']))
    summary = dict(module=r['wasm_sha256'], saved_replies_verified=True, native_bits_equal=True,
                   ordinary_queries=38, cases=cases,
                   scope='Single Q projection; adaptive160/128/32 has no output padding for these tested row counts. Full inference unverified.')
    (D / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    paths = [Path(__file__), ROOT / 'scripts/build_s1_adaptive160_probe.py', ROOT / 'scripts/check_s1_adaptive160_probe.py',
             ROOT / 'scripts/build_s1_output160.py', ROOT / 'scripts/check_s1_output160.py', D / 'frozen-builder.py',
             D / 'upstream.sha256', D / 'frozen-check.py', D / 'check-upstream.sha256', D / 'build/report.json',
             D / 'build/kernel.wat', D / 'build/kernel32.wat', D / 'summary.json'] + list((D / 'build/src').glob('*.rs'))
    with zipfile.ZipFile(D / 'frozen-workflow.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, str(p.relative_to(ROOT)))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
