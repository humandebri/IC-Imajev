#!/usr/bin/env python3
"""Compare current register Delta with per-column stack sums and fused state writes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/delta-stack-v1/build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kernel():
    lines = ['(module (memory 1) (func (export "__imajev_delta_stack") ' + ' '.join(f'(param ${p} i32)' for p in ['q','k','v','g','b','state','out','unused','n']),
             '(local $t i32) (local $qp i32) (local $kp i32) (local $vp i32) (local $op i32) (local $decay v128) (local $beta v128) (local $u v128)']
    lines += [f'(local $s{i}_{j} v128)' for i in range(128) for j in range(32)]
    lines += [f'(local $ki{i} v128) (local $qi{i} v128)' for i in range(128)]
    for i in range(128):
        for j in range(32):
            lines += [f'(local.set $s{i}_{j} (v128.load offset={(i*128+j*4)*4} (local.get $state)))']
    lines += ['(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
    for p in ['q','k','v','out']:
        lines += [f'(local.set ${"op" if p=="out" else p+"p"} (i32.add (local.get ${p}) (i32.shl (local.get $t) (i32.const 9))))']
    for p, name in [('g','decay'),('b','beta')]:
        lines += [f'(local.set ${name} (v128.load32_splat (i32.add (local.get ${p}) (i32.shl (local.get $t) (i32.const 2)))))']
    for i in range(128):
        for p in ['k','q']:
            lines += [f'(local.set ${p}i{i} (v128.load32_splat offset={i*4} (local.get ${p}p)))']
    for j in range(32):
        # V below the positive-zero memory sum: V - sum in original key order.
        lines += ['local.get $vp', f'v128.load offset={j*16}', '(v128.const i32x4 0 0 0 0)']
        for i in range(128):
            lines += [f'(local.tee $s{i}_{j} (f32x4.mul (local.get $s{i}_{j}) (local.get $decay)))', f'local.get $ki{i}', 'f32x4.mul', 'f32x4.add']
        lines += ['f32x4.sub', 'local.get $beta', 'f32x4.mul', 'local.set $u']
        lines += ['local.get $op', '(v128.const i32x4 0 0 0 0)']
        for i in range(128):
            lines += [f'(local.tee $s{i}_{j} (f32x4.add (local.get $s{i}_{j}) (f32x4.mul (local.get $ki{i}) (local.get $u))))', f'local.get $qi{i}', 'f32x4.mul', 'f32x4.add']
        lines += [f'v128.store offset={j*16}']
    lines += ['(local.set $t (i32.add (local.get $t) (i32.const 1))) (br $tokens)))']
    for i in range(128):
        for j in range(32):
            lines += [f'(v128.store offset={(i*128+j*4)*4} (local.get $state) (local.get $s{i}_{j}))']
    lines += ['))']
    return '\n'.join(lines) + '\n'


def main():
    D.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'artifacts/delta-register-v3/build'
    previous = json.loads((original / 'report.json').read_text())
    assert all(sha(ROOT / p) == h for key in ['source_hashes','dependency_hashes'] for p,h in previous[key].items())
    source = (original / 'lib.rs').read_text()
    old = 'delta_simd::run::<false,true>(&q,&k,&v,&g,&b,&mut state,128,128)'
    assert source.count(old) == 1
    call = '__imajev_delta_register(q.as_ptr()as usize,k.as_ptr()as usize,v.as_ptr()as usize,g.as_ptr()as usize,b.as_ptr()as usize,state.as_mut_ptr()as usize,out.as_mut_ptr()as usize,0,n as usize);'
    assert source.count(call) == 1
    source = source.replace(call, call.replace('__imajev_delta_register','__imajev_delta_stack'))
    source = source.replace(old, 'let mut out=vec![0.;n as usize*128];'+call+'out')
    stub = source[source.index('#[unsafe(no_mangle)]'):].replace('__imajev_delta_register','__imajev_delta_stack').replace('(q,k,v,g,b,state,unused,n)', '(q,k,v,g,b,state,unused^517,n)')
    source += '\n' + stub
    (D / 'lib.rs').write_text(source)
    (D / 'delta_simd.rs').write_bytes((original / 'delta_simd.rs').read_bytes())
    (D / 'kernel.wat').write_text(kernel())
    command = previous['command'][:]
    command[command.index('--edition=2021')+1] = str(D / 'lib.rs')
    command[command.index('-o')+1] = str(D / 'raw.wasm')
    env = json.loads((ROOT / 'artifacts/s1_wide/raw128/report.json').read_text())['explicit_env']
    with (D / 'compiler.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, **env), stdout=log, stderr=log, check=True)
    patcher = ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    patches = []
    prior = D / 'raw.wasm'
    for wat, symbol, output in [(original / 'kernel.wat','__imajev_delta_register',D / 'control.wasm'), (D / 'kernel.wat','__imajev_delta_stack',D / 'diagnostic.wasm')]:
        row = json.loads(subprocess.check_output([str(patcher),str(prior),str(wat),str(output),symbol],text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        prior = output
    paths = [Path(__file__), original / 'report.json', original / 'kernel.wat', D / 'lib.rs', D / 'delta_simd.rs', D / 'kernel.wat']
    result = dict(wasm_sha256=sha(prior), source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}, dependency_hashes=previous['dependency_hashes'], command=command, explicit_env=env, patches=patches, scope=__doc__)
    (D / 'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(D / 'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,str(p.relative_to(ROOT)))
    print(result['wasm_sha256'])


if __name__ == '__main__':
    main()
