#!/usr/bin/env python3
"""Build Prefix27 with replacement INT8 kernels and the validated production patches.

Copies frozen optimized sources into a new artifact; never edits the originals,
installs a module or upgrades a canister. Records old/new module and source hashes.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from build_paid_message_checkpoint import projection_bodies, sha
from remove_token_scale import remove_token_scale

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'artifacts/update-rank7-prepare8-unrolled-v1/build'

def replacement_section(text):
    begin = text.index('// Canonical block256 projection;')
    end = text.index('#[cfg(test)]',begin)
    return text[begin:end]

def active_flags(command):
    removed = ('experimental-token-scale','experimental-column','experimental-adopt-column',
               'experimental-balanced44','experimental-adopt-balanced44','experimental-dot-scale',
               'experimental-adopt-dot-scale','experimental-wide-token-tiles','experimental-explicit-weight-loads')
    result=[]
    i=0
    while i<len(command):
        if command[i]=='--cfg' and command[i+1].startswith('feature="') and command[i+1][9:].startswith(removed):
            i+=2
        else:
            result.append(command[i]);i+=1
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True,type=Path)
    args = parser.parse_args()
    directory = (ROOT/args.directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError('artifact must stay inside repository')
    directory.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/build_latest_common_prefix27.py'),
                    '--directory',str(directory/'baseline')],cwd=ROOT,check=True)
    before = json.loads((directory/'baseline/report.json').read_text())
    frozen = json.loads((RUNTIME/'report.json').read_text())
    for group in ('source_hashes','dependency_hashes'):
        for name,digest in frozen[group].items():
            if sha(ROOT/name)!=digest:
                raise ValueError('frozen runtime source/dependency mismatch: '+name)
    baseline_hash = before.get('wasm_sha256',before.get('module'))
    if baseline_hash != sha(directory/'baseline/full.wasm'):
        raise ValueError('baseline module provenance mismatch')
    if 'runtime_command' in before:
        # The current builder also inserts typed token carry and adaptive tiles.
        # Preserve its compiled runtime rather than falling back to the old one.
        for group in ('sources','dependencies'):
            for name,digest in before[group].items():
                if sha(ROOT/name)!=digest:
                    raise ValueError('baseline source/dependency mismatch: '+name)
        runtime_source = directory/'baseline/runtime'
        runtime_command = before['runtime_command'][:]
    else:
        expected = before['dependency_hashes'].get(str((RUNTIME/'libimajev_runtime.rlib').relative_to(ROOT)))
        if expected!=sha(RUNTIME/'libimajev_runtime.rlib'):
            raise ValueError('production runtime provenance mismatch')
        runtime_source = RUNTIME/'runtime'
        runtime_command = frozen['runtime_command'][:]
    shutil.copytree(runtime_source,directory/'runtime')
    owned = ROOT/'crates/imajev-runtime/src'
    path = directory/'runtime/int8_kernel.rs'
    text = path.read_text()
    start=text.index('#[cfg(all(target_arch="wasm32",feature="experimental-dot-scale"))]')
    end=text.index('#[cfg(test)]',start)
    text=text[:start]+replacement_section((owned/'int8_kernel.rs').read_text())+text[end:]
    path.write_text(text.replace('#[cfg(all(test, feature = "experimental-wide-token-tiles"))]','#[cfg(test)]'))
    for name in ('int8_dot_scale.rs','int8_column32.rs','int8_token_kernel.rs'):
        (directory/'runtime'/name).unlink()
    shutil.copyfile(owned/'int8_tile.rs',directory/'runtime/int8_tile.rs')
    lib = directory/'runtime/lib.rs'
    text = remove_token_scale(lib.read_text())
    if text.count('pub mod int8_kernel;')!=1:
        raise ValueError('runtime module anchor mismatch')
    lib.write_text(text.replace('pub mod int8_kernel;','#[cfg(target_arch="wasm32")]\nmod int8_tile;\npub mod int8_kernel;',1))
    sources = list((directory/'runtime').rglob('*.rs'))+[Path(__file__),ROOT/'scripts/generate_int8_tile.py',ROOT/'scripts/remove_token_scale.py']+[owned/name for name in ('int8_kernel.rs','int8_tile.rs','lib.rs')]
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    command = active_flags(runtime_command)
    command[command.index('--edition=2021')+1] = str(lib)
    command[command.index('-o')+1] = str(directory/'libimajev_runtime.rlib')
    with (directory/'runtime-compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=log)
    wrapper = active_flags(before['command'])
    wrapper = [f'imajev_runtime={directory}/libimajev_runtime.rlib' if value.startswith('imajev_runtime=') else value for value in wrapper]
    wrapper[wrapper.index('-o')+1] = str(directory/'raw.wasm')
    env = dict(os.environ,CARGO_MANIFEST_DIR=str(directory/'baseline'),CARGO_PKG_NAME='imajev-inference',
               CARGO_PKG_VERSION='0.1.0',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',
               CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_inference')
    with (directory/'compiler.log').open('w') as log:
        subprocess.run(wrapper,cwd=ROOT,env=env,check=True,stdout=log,stderr=log)
    parent = ROOT/'artifacts/paid-stack-carry-projection-v1/build'
    patches = json.loads((parent/'report.json').read_text())['patches']
    donors = projection_bodies((parent/'full.wasm').read_bytes(),patches)
    previous,applied = directory/'raw.wasm',[]
    for index,(patch,body) in enumerate(zip(patches,donors,strict=True)):
        donor = directory/f'donor-{index}.wasm'
        donor.write_bytes(body)
        output = directory/('full.wasm' if index==len(patches)-1 else f'patched-{index}.wasm')
        result = json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),
                                                    str(previous),str(donor),str(output),patch['export']],text=True))
        if not result['wasmparser_validation'] or result['replacement_body_sha256']!=patch['replacement_body_sha256']:
            raise ValueError('production projection patch mismatch')
        applied.append(result)
        previous = output
    if source_hashes!={str(p.relative_to(ROOT)):sha(p) for p in sources}:
        raise ValueError('replacement sources changed during compilation')
    report = dict(complete=True,scope='Build only; no new full-model inference or deployment',
                  baseline_wasm_sha256=baseline_hash,wasm_sha256=sha(directory/'full.wasm'),
                  identical_module=baseline_hash==sha(directory/'full.wasm'),
                  baseline_report_sha256=sha(directory/'baseline/report.json'),
                  input_limit=before.get('input_limit',116),
                  adaptive_token_tiles=before.get('adaptive_token_tiles',False),
                  runtime_command=command,command=wrapper,source_hashes=source_hashes,
                  frozen_runtime_report_sha256=sha(RUNTIME/'report.json'),dependency_hashes=frozen['dependency_hashes'],
                  patches=applied,all34_projection_bodies_equal_to_validated_parent=len(applied)==34,
                  full_inference_verified=False)
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(module=report['wasm_sha256'],identical_module=report['identical_module'],patches=len(applied))))

if __name__=='__main__':
    import sys
    main()
