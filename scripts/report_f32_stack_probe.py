#!/usr/bin/env python3
"""Re-decode stack-accumulator evidence and compare identical output128 fixtures."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/report_f32_output128_probe.py'
    source = p.read_text().replace('artifacts/f32-output128-v1', 'artifacts/f32-stack-v1').replace('scripts/check_f32_output128_probe.py', 'scripts/check_f32_stack_probe.py')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))
    d = ROOT / 'artifacts/f32-stack-v1'
    current = json.loads((d / 'down-B/report.json').read_text())
    previous_path = ROOT / 'artifacts/f32-output128-v1/down-B/report.json'
    previous = json.loads(previous_path.read_text())
    assert current['weight_sha256'] == previous['weight_sha256']
    assert current['model'] == previous['model'] and current['pack_hash'] == previous['pack_hash']
    cases = []
    for a, b in zip(current['cases'], previous['cases'], strict=True):
        assert a['label'] == b['label'] and a['input_sha256'] == b['input_sha256']
        assert a['native']['digest'] == b['native']['digest']
        before = b['measurements']['output128']['total_instructions']
        after = a['measurements']['output128']['total_instructions']
        cases.append(dict(label=a['label'], tokens=a['tokens'], previous_output128=before,
                          stack_output128=after, reduction_percent=100 * (1 - after / before)))
    result = dict(native_bits_equal=True, saved_replies_verified=True,
                  previous_report_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                  current_report_sha256=hashlib.sha256((d / 'down-B/report.json').read_bytes()).hexdigest(),
                  cases=cases, scope='Single down-B projection only; whole-update inference remains unproven.')
    (d / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    files = [Path(__file__), ROOT / 'scripts/build_f32_stack_probe.py', ROOT / 'scripts/check_f32_stack_probe.py',
             ROOT / 'scripts/build_f32_output128_probe.py', ROOT / 'scripts/build_f32_output64_probe.py',
             ROOT / 'scripts/check_f32_output128_probe.py', p, d / 'build/report.json', d / 'build/wide.wat',
             d / 'frozen-check.py', d / 'check-upstream.sha256', d / 'summary.json'] + list((d / 'build/src').glob('*.rs'))
    with zipfile.ZipFile(d / 'frozen-workflow.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, str(path.relative_to(ROOT)))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
