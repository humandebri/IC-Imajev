#!/usr/bin/env python3
"""Separate exact rank1127 query input preparation from kernel loop counters."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = ROOT/'artifacts/rank1127-generic-prep-v1/build'
    prior = json.loads((old/'report.json').read_text())
    for key in ('source_hashes','dependency_hashes'):
        assert all(sha(ROOT/p)==h for p,h in prior[key].items())
    d = ROOT/'artifacts/rank1127-core-profile-v1/build'
    d.mkdir(parents=True,exist_ok=False)
    src = d/'src'
    shutil.copytree(old/'src',src)
    shutil.copyfile(old/'kernel.wat',d/'kernel.wat')
    p = src/'exact.rs'
    source = p.read_text()
    source = 'std::thread_local!{static PHASE:std::cell::Cell<(u64,u64)>=const{std::cell::Cell::new((0,0))};}\npub fn phases()->(u64,u64){PHASE.with(|v|v.get())}\n'+source
    replacements = [
        ('let padded=rows.div_ceil(12)*12;','PHASE.with(|v|v.set((0,0)));let padded=rows.div_ceil(12)*12;'),
        ('let operands=crate::prepare_s3::input(q,block);','let begin=ic_cdk::api::performance_counter(0);let operands=crate::prepare_s3::input(q,block);let prepared=ic_cdk::api::performance_counter(0);'),
        ('q.rows());}}','q.rows());}let finished=ic_cdk::api::performance_counter(0);PHASE.with(|v|{let(a,b)=v.get();v.set((a+prepared-begin,b+finished-prepared));});}')]
    for before,after in replacements:
        assert source.count(before)==1,before
        source = source.replace(before,after)
    p.write_text(source)
    p = src/'lib.rs'
    source = p.read_text()
    before = 'input_prepare_instructions:prep-quant,project_instructions:end-prep'
    after = 'input_prepare_instructions:if method==4{exact::phases().0}else{prep-quant},project_instructions:if method==4{exact::phases().1}else{end-prep}'
    assert source.count(before)==1
    p.write_text(source.replace(before,after))
    command = prior['command'][:]
    command[command.index('--edition=2021')+1] = str(src/'lib.rs')
    command[command.index('-o')+1] = str(d/'raw.wasm')
    with (d/'compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**prior['explicit_env']),stdout=log,stderr=log,check=True)
    control = ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    previous = d/'raw.wasm'
    patches = []
    for file,symbol,output in ((control,'__imajev_s1_wide_accumulate',d/'control.wasm'),(d/'kernel.wat','__imajev_s3_stream_accumulate',d/'diagnostic.wasm')):
        row = json.loads(subprocess.check_output([str(patcher),str(previous),str(file),str(output),symbol],text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        previous = output
    paths = [Path(__file__),old/'report.json',old/'kernel.wat',d/'kernel.wat',control]+list(src.glob('*.rs'))
    hashes = dict(prior['source_hashes'])
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    result = dict(prior,wasm_sha256=sha(previous),source_hashes=hashes,command=command,patches=patches,scope=__doc__,profiling_only=True,query_kernel_byte_equal_to_expanded=True)
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(d/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in hashes:
            z.write(ROOT/path,path)
    print(json.dumps({'module':result['wasm_sha256'],'locals':8685}))


if __name__=='__main__':
    main()
