#!/usr/bin/env python3
"""Build a diagnostic of the frozen update runtime; preserve every production kernel."""
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
D = ROOT / 'artifacts/update-instruction-profile-v1/build-v3'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    D.mkdir(parents=True, exist_ok=False)
    base = json.loads((B/'report.json').read_text())
    update = json.loads((U/'report.json').read_text())
    for report in [base, update]:
        for key in ['source_hashes', 'dependency_hashes']:
            assert all(sha(ROOT/p) == h for p, h in report[key].items()), key
    shutil.copytree(B/'runtime', D/'runtime')
    for p in U.glob('*.rs'):
        shutil.copyfile(p, D/p.name)
    p = D/'update_inference.rs'
    s = p.read_text()
    s += '''
thread_local! {static LAST_PROFILE: RefCell<Vec<(String,u64,u64)>> = RefCell::new(Vec::new());}
#[ic_cdk::query]
fn last_instruction_profile() -> String {
    owner();
    LAST_PROFILE.with(|p|serde_json::to_string(&*p.borrow()).unwrap())
}
'''
    for signature in [
        'fn update_infer_start(ids: Vec<u32>, options: Vec<String>) -> Result<UpdateProgress,String> {',
        'fn update_infer_continue(id:u64, stage:u64) -> Result<UpdateProgress,String> {',
    ]:
        assert s.count(signature) == 1
        s = s.replace(signature, signature+'\n    imajev_runtime::profile::start(||ic_cdk::api::performance_counter(0));')
    anchor = 'GRAPH.with(|g|g.borrow_mut().session=Some(s));reply'
    assert s.count(anchor) == 1
    s = s.replace(anchor, 'LAST_PROFILE.with(|p|*p.borrow_mut()=imajev_runtime::profile::finish());\n    '+anchor)
    p.write_text(s)
    runtime = base['runtime_command'][:]
    runtime[runtime.index('--edition=2021')+1] = str(D/'runtime/lib.rs')
    runtime[runtime.index('-o')+1] = str(D/'libimajev_runtime_profile.rlib')
    runtime += ['--cfg', 'feature="instruction-profile"']
    wrapper = update['command'][:]
    wrapper[wrapper.index('--edition=2021')+1] = str(D/'lib.rs')
    wrapper[wrapper.index('-o')+1] = str(D/'raw.wasm')
    for i, arg in enumerate(wrapper):
        if arg.startswith('imajev_runtime='):
            wrapper[i] = 'imajev_runtime='+str(D/'libimajev_runtime_profile.rlib')
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(D), CARGO_PKG_NAME='imajev-runtime', CARGO_PKG_VERSION='0.1.0')
    with (D/'runtime-compiler.log').open('w') as log:
        subprocess.run(runtime, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    env.update(CARGO_PKG_NAME='imajev-inference', CARGO_PKG_VERSION_MAJOR='0', CARGO_PKG_VERSION_MINOR='1', CARGO_PKG_VERSION_PATCH='0', CARGO_PKG_VERSION_PRE='', CARGO_CRATE_NAME='imajev_inference')
    with (D/'compiler.log').open('w') as log:
        subprocess.run(wrapper, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    previous = D/'raw.wasm'
    patches = []
    for i, patch in enumerate(base['patches']):
        wat = ROOT/'artifacts/single-quad/build-v2'/f'kernel{i}.wat'
        if i == 3:
            wat = ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
        if i == 4:
            wat = ROOT/'artifacts/delta-register-v3/build/kernel.wat'
        if i == 5:
            wat = ROOT/'artifacts/f32-output64-v1/build-v2/wide.wat'
        assert sha(wat) == patch['source_sha256']
        out = D/('full.wasm' if i == 5 else f'patched{i}.wasm')
        row = json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'), str(previous), str(wat), str(out), patch['export']], text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        previous = out
    sources = [Path(__file__), B/'report.json', U/'report.json']+list(D.glob('*.rs'))+list((D/'runtime').glob('*.rs'))
    report = dict(baseline=base['baseline'], wasm_sha256=sha(D/'full.wasm'), source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}, dependency_hashes=base['dependency_hashes'], runtime_command=runtime, command=wrapper, patches=patches, scope='Diagnostic inclusive spans; production kernels unchanged; never a production improvement measurement.')
    (D/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, str(p.relative_to(ROOT)))
    print(json.dumps(dict(module=report['wasm_sha256'], bytes=(D/'full.wasm').stat().st_size)))


if __name__ == '__main__':
    main()
