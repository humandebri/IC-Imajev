#!/usr/bin/env python3
"""Build isolated W4 diagnostic with frozen, already compiled CDK dependencies."""
import argparse, hashlib, json, os, pathlib, subprocess, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);ap.add_argument('--source-directory',default='scripts/w4_lut_bench/src');a=ap.parse_args()
    d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
    if any(d.iterdir()):raise ValueError('Use a fresh build directory')
    cached=json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text())
    cmd=cached['command'];deps={cmd[i+1].split('=',1)[0]:pathlib.Path(cmd[i+1].split('=',1)[1]) for i,v in enumerate(cmd) if v=='--extern'}
    deps.pop('imajev_runtime')
    source=ROOT/a.source_directory/'lib.rs'
    sources=list(source.parent.glob('*.rs'))+[pathlib.Path(__file__)]
    generator=ROOT/'scripts/generate_w4_lut_wide.py'
    if generator.exists():sources.append(generator)
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in deps.values()}
    command=['rustc','--crate-name','imajev_w4_lut_bench','--edition=2021',str(source),'--crate-type','cdylib','--target','wasm32-unknown-unknown','-C','opt-level=3','-C','target-feature=+simd128','-C','panic=abort','-C','codegen-units=1','-C','lto=thin','-C','overflow-checks=yes','--emit=link','-o',str(d/'diagnostic.wasm')]
    command+=['-L',f'dependency={ROOT/"target/wasm32-unknown-unknown/release/deps"}','-L',f'dependency={ROOT/"target/release/deps"}']
    for name,p in deps.items():command+=['--extern',f'{name}={p}']
    env=dict(os.environ);env.update(CARGO_MANIFEST_DIR=str(source.parent.parent),CARGO_PKG_NAME='imajev-w4-lut-bench',CARGO_PKG_VERSION='0.1.0',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_w4_lut_bench')
    with (d/'compiler.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert source_hashes=={str(p.relative_to(ROOT)):sha(p) for p in sources}
    assert dependency_hashes=={str(p.relative_to(ROOT)):sha(p) for p in deps.values()}
    with zipfile.ZipFile(d/'source.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in sources:z.write(p,str(p.relative_to(ROOT)))
    result=dict(command=command,compiler=subprocess.check_output(['rustc','--version','--verbose'],text=True),source_hashes=source_hashes,dependency_hashes=dependency_hashes,wasm_sha256=sha(d/'diagnostic.wasm'),wasm_bytes=(d/'diagnostic.wasm').stat().st_size)
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
