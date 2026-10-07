#!/usr/bin/env python3
"""Re-decode each columns512 reply and freeze current-stack comparison proof."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = ROOT / 'scripts/report_f32_output128_probe.py'
    source = p.read_text().replace('artifacts/f32-output128-v1', 'artifacts/f32-columns512-v1')
    source = source.replace('scripts/check_f32_output128_probe.py', 'scripts/check_f32_columns512_probe.py')
    source = source.replace("['down-B']", "['gate-A']")
    source = source.replace('output32', 'stack64').replace('output128', 'columns512')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))
    d = ROOT / 'artifacts/f32-columns512-v1'
    assert sha(d/'build/wide.wat') == sha(ROOT/'artifacts/update-f32-delta-stack-v1/wide64.wat')
    r = json.loads((d/'report.json').read_text())
    files = [Path(__file__), ROOT/'scripts/build_f32_columns512_probe.py', ROOT/'scripts/check_f32_columns512_probe.py',
             ROOT/'scripts/build_f32_output64_probe.py', ROOT/'scripts/build_f32_stack_probe.py',
             ROOT/'scripts/check_f32_output128_probe.py', p, d/'build/report.json', d/'build/wide.wat',
             d/'build/columns512.wat', d/'frozen-builder.py', d/'frozen-kernel-generator.py',
             d/'frozen-check.py', d/'upstream.sha256', d/'check-upstream.sha256']
    files += list((d/'build/src').glob('*.rs'))
    result = dict(module=r['wasm_sha256'], native_bits_equal=True, saved_replies_verified=True,
                  current_stack64_wat_identical=True, cases=r['tensors'][0]['reductions'],
                  workflow_hashes={str(f.relative_to(ROOT)):sha(f) for f in files},
                  scope='Single gate-A projection only; entire update inference unproven for columns512.')
    (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    files.append(d/'summary.json')
    with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f,str(f.relative_to(ROOT)))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
