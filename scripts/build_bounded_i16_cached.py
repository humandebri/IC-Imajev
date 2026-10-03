#!/usr/bin/env python3
"""Build the isolated diagnostic against explicitly pinned compiled dependencies.

This avoids rebuilding the full inference runtime for an unused adopt flag.
Records exact source/dependency hashes, flags, compiler and explicit Cargo env.
Does not install a canister or modify Cargo's target directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', required=True)
    ap.add_argument('--directory', default='artifacts/bounded_i16/cached-build')
    args = ap.parse_args()
    dest = ROOT / args.directory
    dest.mkdir(parents=True, exist_ok=True)
    deps = ROOT / 'target/wasm32-unknown-unknown/release/deps'
    runtime = (ROOT / args.runtime).resolve()
    paths = {'imajev_runtime': runtime}
    for name in ['serde','sha2','ic_cdk','candid']:
        choices = list(deps.glob(f'lib{name}-*.rlib'))
        if len(choices) != 1:
            raise ValueError(f'Expected one pinned {name} dependency; found {len(choices)}')
        paths[name] = choices[0]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source = ROOT / 'scripts/bounded_i16_bench'
    sources = list((source/'src').rglob('*.rs')) + [source/'Cargo.toml',source/'Cargo.lock',
        ROOT/'scripts/generate_bounded_i16.py',Path(__file__)]
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    dependency_hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    runtime_fingerprint = ROOT/'target/wasm32-unknown-unknown/release/.fingerprint'/runtime.name.removeprefix('lib').removesuffix('.rlib').replace('imajev_runtime-','imajev-runtime-')/'lib-imajev_runtime.json'
    fingerprint = json.loads(runtime_fingerprint.read_text())
    assert 'experimental-prepared-output-pairs' in fingerprint['features']
    assert 'experimental-column32' in fingerprint['features']
    output = dest/'diagnostic.wasm'
    cmd = ['rustc','--crate-name','imajev_bounded_i16_bench','--edition=2021',str(source/'src/lib.rs'),
        '--crate-type','cdylib','--target','wasm32-unknown-unknown',
        '-C','opt-level=3','-C','panic=abort','-C','codegen-units=1','-C','lto=thin',
        '-C','overflow-checks=yes','--emit=link','-o',str(output),
        '-L',f'dependency={deps}','-L',f'dependency={ROOT/"target/release/deps"}']
    for name,path in paths.items():
        cmd += ['--extern',f'{name}={path}']
    explicit_env = {'CARGO_MANIFEST_DIR':str(source),'CARGO_PKG_NAME':'imajev-bounded-i16-bench',
        'CARGO_PKG_VERSION':'0.1.0','CARGO_PKG_VERSION_MAJOR':'0','CARGO_PKG_VERSION_MINOR':'1',
        'CARGO_PKG_VERSION_PATCH':'0','CARGO_PKG_VERSION_PRE':'','CARGO_CRATE_NAME':'imajev_bounded_i16_bench'}
    env = dict(os.environ)
    env.update(explicit_env)
    compiler = subprocess.check_output(['rustc','--version','--verbose'],text=True)
    with (dest/'compiler.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert source_hashes == {str(p.relative_to(ROOT)):sha(p) for p in sources}
    assert dependency_hashes == {str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    result = dict(compiler=compiler,command=cmd,explicit_env=explicit_env,
        source_hashes=source_hashes,dependency_hashes=dependency_hashes,
        runtime_features=fingerprint['features'],wasm_sha256=sha(output),wasm_bytes=output.stat().st_size,
        scope='Diagnostic only; uses the pinned prepared-pairs runtime already compiled by Cargo. No canister install.')
    (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['wasm_sha256','wasm_bytes']}))

if __name__ == '__main__':
    main()
