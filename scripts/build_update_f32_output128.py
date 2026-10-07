#!/usr/bin/env python3
"""Build unprofiled update inference with adaptive exact F32 output128/64 tiles."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'artifacts/voting-template-prefix-v1/full-build'
U = ROOT / 'artifacts/update-templates-v1/build'
D = ROOT / 'artifacts/update-f32-output128-v1/build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    D.mkdir(parents=True, exist_ok=False)
    base = json.loads((B / 'report.json').read_text())
    update = json.loads((U / 'report.json').read_text())
    for report in [base, update]:
        for key in ['source_hashes', 'dependency_hashes']:
            assert all(sha(ROOT / p) == h for p, h in report[key].items())
    shutil.copytree(B / 'runtime', D / 'runtime')
    for p in U.glob('*.rs'):
        shutil.copyfile(p, D / p.name)
    p = D / 'runtime/f32_output.rs'
    s = p.read_text()
    start = s.index('// Original packed output32 stride;')
    clone = s[start:].replace('project_wide(', 'project_wide128(')
    clone = clone.replace('rows%64', 'rows%128').replace('output64', 'output128')
    clone = clone.replace('step_by(64) {\n  let global', 'step_by(128) {\n  let global')
    clone = clone.replace('(global+64)*cols', '(global+128)*cols')
    clone = clone.replace('[[0f32;64]', '[[0f32;128]')
    clone = clone.replace('accumulate_wide(', 'accumulate_wide128(')
    clone = clone.replace('x.as_ptr(),64,', 'x.as_ptr(),128,')
    clone = clone.replace('r+64]', 'r+128]').replace('__imajev_f32_wide"', '__imajev_f32_wide128"')
    clone = clone.replace('0..n*64', '0..n*128').replace('(marker^5)', '(marker^517)')
    signature = 'fn project_wide(x:&[f32],packed:&[f32],start:usize,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {'
    assert s.count(signature) == 1
    s = s.replace(signature, signature + '\n if rows%128==0 {return project_wide128(x,packed,start,n,rows,cols);}')
    p.write_text(s + '\n' + clone)
    runtime = base['runtime_command'][:]
    runtime[runtime.index('--edition=2021') + 1] = str(D / 'runtime/lib.rs')
    runtime[runtime.index('-o') + 1] = str(D / 'libimajev_runtime.rlib')
    wrapper = update['command'][:]
    wrapper[wrapper.index('--edition=2021') + 1] = str(D / 'lib.rs')
    wrapper[wrapper.index('-o') + 1] = str(D / 'raw.wasm')
    for i, arg in enumerate(wrapper):
        if arg.startswith('imajev_runtime='):
            wrapper[i] = 'imajev_runtime=' + str(D / 'libimajev_runtime.rlib')
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(D), CARGO_PKG_NAME='imajev-runtime', CARGO_PKG_VERSION='0.1.0')
    with (D / 'runtime-compiler.log').open('w') as log:
        subprocess.run(runtime, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    env.update(CARGO_PKG_NAME='imajev-inference', CARGO_PKG_VERSION_MAJOR='0', CARGO_PKG_VERSION_MINOR='1', CARGO_PKG_VERSION_PATCH='0', CARGO_PKG_VERSION_PRE='', CARGO_CRATE_NAME='imajev_inference')
    with (D / 'compiler.log').open('w') as log:
        subprocess.run(wrapper, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    previous = D / 'raw.wasm'
    patches = []
    for i, patch in enumerate(base['patches']):
        wat = ROOT / 'artifacts/single-quad/build-v2' / f'kernel{i}.wat'
        if i == 3:
            wat = ROOT / 'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
        if i == 4:
            wat = ROOT / 'artifacts/delta-register-v3/build/kernel.wat'
        if i == 5:
            wat = ROOT / 'artifacts/f32-output64-v1/build-v2/wide.wat'
        assert sha(wat) == patch['source_sha256']
        out = D / f'patched{i}.wasm'
        row = json.loads(subprocess.check_output([str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'), str(previous), str(wat), str(out), patch['export']], text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        previous = out
    wat = D / 'wide128.wat'
    original = ROOT / 'artifacts/f32-output128-v1/build/wide.wat'
    assert sha(original) == json.loads((original.parent / 'report.json').read_text())['source_hashes'][str(original.relative_to(ROOT))]
    wat.write_text(original.read_text().replace('__imajev_f32_wide"', '__imajev_f32_wide128"'))
    row = json.loads(subprocess.check_output([str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'), str(previous), str(wat), str(D / 'full.wasm'), '__imajev_f32_wide128'], text=True))
    assert row['wasmparser_validation']
    patches.append(row)
    sources = [Path(__file__), B / 'report.json', U / 'report.json', original, wat] + list(D.glob('*.rs')) + list((D / 'runtime').glob('*.rs'))
    report = dict(baseline=base['baseline'], wasm_sha256=sha(D / 'full.wasm'), source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources}, dependency_hashes=base['dependency_hashes'], runtime_command=runtime, command=wrapper, patches=patches, scope=__doc__)
    (D / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with zipfile.ZipFile(D / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, str(p.relative_to(ROOT)))
    print(json.dumps(dict(module=report['wasm_sha256'], bytes=(D / 'full.wasm').stat().st_size)))


if __name__ == '__main__':
    main()
