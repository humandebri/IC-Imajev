#!/usr/bin/env python3
"""Re-decode output512 proof and compare identical stack-output128 fixtures."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = ROOT / 'scripts/report_f32_output128_probe.py'
    source = p.read_text().replace('artifacts/f32-output128-v1', 'artifacts/f32-stack512-v1')
    source = source.replace('scripts/check_f32_output128_probe.py', 'scripts/check_f32_stack512_probe.py')
    source = source.replace('output128', 'output512')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))
    d = ROOT / 'artifacts/f32-stack512-v1'
    current_path = d / 'down-B/report.json'
    previous_path = ROOT / 'artifacts/f32-stack-v1/down-B/report.json'
    current, previous = json.loads(current_path.read_text()), json.loads(previous_path.read_text())
    assert all(current[k] == previous[k] for k in ['weight_sha256', 'model', 'pack_hash', 'tensor'])
    cases = []
    for a, b in zip(current['cases'], previous['cases'], strict=True):
        assert all(a[k] == b[k] for k in ['label', 'tokens', 'rows', 'cols', 'input_sha256'])
        assert a['native']['digest'] == b['native']['digest']
        before = b['measurements']['output128']['total_instructions']
        after = a['measurements']['output512']['total_instructions']
        cases.append(dict(label=a['label'], tokens=a['tokens'], previous_stack128=before,
                          stack512=after, reduction_percent=100 * (1 - after / before)))
    files = [Path(__file__), ROOT / 'scripts/build_f32_stack512_probe.py',
             ROOT / 'scripts/check_f32_stack512_probe.py', ROOT / 'scripts/build_f32_stack_probe.py',
             ROOT / 'scripts/build_f32_output128_probe.py', ROOT / 'scripts/build_f32_output64_probe.py',
             ROOT / 'scripts/check_f32_output128_probe.py', p, d / 'build/report.json',
             d / 'build/wide.wat', d / 'frozen-builder.py', d / 'frozen-kernel-generator.py',
             d / 'frozen-check.py', d / 'upstream.sha256', d / 'check-upstream.sha256']
    files += list((d / 'build/src').glob('*.rs'))
    result = dict(module=current['wasm_sha256'], native_bits_equal=True, saved_replies_verified=True,
                  previous_report_sha256=sha(previous_path), current_report_sha256=sha(current_path),
                  workflow_hashes={str(f.relative_to(ROOT)): sha(f) for f in files}, cases=cases,
                  scope='Single down-B projection only; full update inference has not used output512.')
    (d / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    files.append(d / 'summary.json')
    with zipfile.ZipFile(d / 'frozen-workflow.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, str(f.relative_to(ROOT)))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
