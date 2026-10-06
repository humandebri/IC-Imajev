#!/usr/bin/env python3
"""Compile the generated S3 probe and latest exact S1 control in one module."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/s3-stream-v1/build'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    (D/'src').mkdir(exist_ok=False)
    source=ROOT/'scripts/s1_wide_bench/src'
    lib=(source/'lib.rs').read_text()
    lib=lib.replace('pub mod exact;','pub mod exact;\n#[cfg(target_arch="wasm32")]mod prepare_s3;\n#[cfg(target_arch="wasm32")]mod s3_kernel;')
    lib=lib.replace('assert!(method<=3)','assert!(method==3||method==4)')
    lib=lib.replace('(method>0).then(', '(method==3).then(')
    lib=lib.replace('let out=if method==3 {','let out=if method==4 {f.coeff.as_ref().unwrap().project_s3(&q,&f.scales[..rows],rows).unwrap()}else if method==3 {')
    (D/'src/lib.rs').write_text(lib)
    exact=(source/'exact.rs').read_text().replace('impl Prepared {','impl Prepared {\n'+(D/'project.rs').read_text(),1)
    (D/'src/exact.rs').write_text(exact)
    (D/'src/kernel.rs').write_bytes((source/'kernel.rs').read_bytes())
    for name in ['prepare_s3.rs','s3_kernel.rs']:(D/'src'/name).write_bytes((D/name).read_bytes())
    old=json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text())
    command=old['command'][:];command[command.index('--edition=2021')+1]=str(D/'src/lib.rs')
    command[command.index('-o')+1]=str(D/'raw.wasm')
    control=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
    control_patch=json.loads((ROOT/'artifacts/s1-pair-bounds-v1/build/full.patch.json').read_text())
    assert sha(control)==control_patch['source_sha256']
    sources=list((D/'src').glob('*.rs'))+[D/'kernel.wat',D/'generator.json',control,Path(__file__),ROOT/'scripts/generate_s3_stream_probe.py']
    identities={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for path,digest in old['dependency_hashes'].items():assert sha(ROOT/path)==digest
    with (D/'compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),check=True,stdout=log,stderr=log)
    patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch';patches=[]
    for previous,wat,out,symbol in [(D/'raw.wasm',control,D/'control.wasm','__imajev_s1_wide_accumulate'),
                                    (D/'control.wasm',D/'kernel.wat',D/'diagnostic.wasm','__imajev_s3_stream_accumulate')]:
        patches.append(json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True)))
    assert identities=={p:sha(ROOT/p) for p in identities}
    for path,digest in old['dependency_hashes'].items():assert sha(ROOT/path)==digest
    report=dict(command=command,explicit_env=old['explicit_env'],source_hashes=identities,
                dependency_hashes=old['dependency_hashes'],patcher_sha256=sha(patcher),patches=patches,
                wasm_sha256=sha(D/'diagnostic.wasm'),control_module=control_patch['output_sha256'],scope=__doc__)
    (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:archive.write(path,str(path.relative_to(ROOT)))
    print(json.dumps(dict(wasm_sha256=report['wasm_sha256'],wasm_bytes=(D/'diagnostic.wasm').stat().st_size)))


if __name__=='__main__':main()
