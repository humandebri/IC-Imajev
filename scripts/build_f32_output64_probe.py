#!/usr/bin/env python3
"""Build an isolated output32/output64 LoRA comparison on the same packed weights."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/f32-output64-v1/build'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel64():
    lines = ['(module (func (export "__imajev_f32_wide") ' +
             ' '.join(f'(param ${p} i32)' for p in ['q','w','cols','start','sx','stride','sw','sums','n']),
             '(local $t i32) (local $qp i32) (local $yp i32) (local $wp i32) (local $wp1 i32) (local $x v128)']
    lines += [f'(local $s{r} v128)' for r in range(16)]
    lines += [f'(local $w{c}_{r} v128)' for c in range(64) for r in range(16)]

    def token(first):
        out = ['(local.set $wp (i32.add (local.get $w) (i32.shl (local.get $start) (i32.const 7))))',
               '(local.set $wp1 (i32.add (local.get $wp) (i32.shl (local.get $cols) (i32.const 7))))',
               '(local.set $qp (i32.add (local.get $q) (i32.shl (i32.add (i32.mul (local.get $t) (local.get $cols)) (local.get $start)) (i32.const 2))))',
               '(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2))))']
        out += [f'(local.set $s{r} (v128.load offset={r*16} (local.get $yp)))' for r in range(16)]
        for c in range(64):
            for r in range(16):
                x = f'(local.tee $x (v128.load32_splat offset={c*4} (local.get $qp)))' if r == 0 else '(local.get $x)'
                pointer = '$wp' if r < 8 else '$wp1'
                weight = (f'(local.tee $w{c}_{r} (v128.load offset={c*128+(r%8)*16} (local.get {pointer})))'
                          if first else f'(local.get $w{c}_{r})')
                out.append(f'(local.set $s{r} (f32x4.add (local.get $s{r}) (f32x4.mul {x} {weight})))')
        out += [f'(v128.store offset={r*16} (local.get $yp) (local.get $s{r}))' for r in range(16)]
        return out

    lines += ['(local.set $t (i32.const 0))', '(block $done (br_if $done (i32.eqz (local.get $n)))']
    lines += token(True)
    lines += ['(local.set $t (i32.const 1))', '(loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
    lines += token(False)
    lines += ['(local.set $t (i32.add (local.get $t) (i32.const 1)))', 'br $tokens', '))', '))']
    return '\n'.join(lines) + '\n'


def main():
    global D
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='artifacts/f32-output64-v1/build-v2')
    D = ROOT / parser.parse_args().directory
    D.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'scripts/f32_block_bench/src'
    (D/'src').mkdir()
    lib = (source/'lib.rs').read_text()
    lib = lib.replace('rows%32==0', 'rows%64==0')
    old = 'imajev_runtime::matrix(&x,&f.w,n,f.rows,f.cols).unwrap()'
    assert old in lib
    lib = lib.replace(old, 'f.packed.as_ref().unwrap().project(&x,n).unwrap()')
    lib = lib.replace('else{f.packed.as_ref().unwrap().project(&x,n).unwrap()}',
                      'else{f.packed.as_ref().unwrap().project64(&x,n).unwrap()}')
    (D/'src/lib.rs').write_text(lib)
    packed = (source/'packed.rs').read_text().split('#[cfg(test)]')[0]
    begin = packed.index(' pub fn project(')
    original = packed[begin:packed.rfind('\n}')]
    wide = original.replace('pub fn project(', 'pub fn project64(')
    wide = wide.replace('step_by(32)', 'step_by(64)').replace('[[0f32;32]', '[[0f32;64]')
    wide = wide.replace('crate::kernel::accumulate(', 'crate::kernel::accumulate64(')
    wide = wide.replace('x.as_ptr(),32,', 'x.as_ptr(),64,').replace('r+32]', 'r+64]')
    (D/'src/packed.rs').write_text(packed[:packed.rfind('\n}')] + '\n' + wide + '\n}\n')
    stub = (source/'kernel.rs').read_text()
    second = stub.replace('//!','//').replace('__imajev_pair_accumulate','__imajev_f32_wide').replace('fn accumulate(', 'fn accumulate64(').replace('n*32','n*64')
    (D/'src/kernel.rs').write_text(stub + '\n' + second)
    baseline = ROOT/'artifacts/f32_block/build/kernel.wat'
    adopted_patch = json.loads((ROOT/'artifacts/query-packing-v3/build/patch2.json').read_text())
    assert sha(baseline) == adopted_patch['source_sha256']
    (D/'wide.wat').write_text(kernel64())
    old_build = json.loads((ROOT/'artifacts/f32_block/cached-build/report.json').read_text())
    command = old_build['command'][:]
    command[command.index('--crate-name')+1] = 'imajev_f32_output64_probe'
    command[command.index('--edition=2021')+1] = str(D/'src/lib.rs')
    command[command.index('-o')+1] = str(D/'raw.wasm')
    dependencies = [pathlib.Path(command[i+1].split('=',1)[1])
                    for i, value in enumerate(command) if value == '--extern']
    sources = list((D/'src').glob('*.rs')) + [D/'wide.wat', baseline, pathlib.Path(__file__)]
    identities = {str(p.relative_to(ROOT)):sha(p) for p in sources+dependencies}
    environment = dict(os.environ, **old_build['explicit_env'])
    with (D/'compiler.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=environment, check=True, stdout=log, stderr=log)
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'
    patches = []
    for previous, wat, output, symbol in [
        (D/'raw.wasm',baseline,D/'control.wasm','__imajev_pair_accumulate'),
        (D/'control.wasm',D/'wide.wat',D/'diagnostic.wasm','__imajev_f32_wide')]:
        patches.append(json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(output),symbol], text=True)))
    assert identities == {str(p.relative_to(ROOT)):sha(p) for p in sources+dependencies}
    result = dict(command=command, explicit_env=old_build['explicit_env'], source_hashes=identities,
                  patches=patches, wasm_sha256=sha(D/'diagnostic.wasm'), patcher_sha256=sha(patcher),
                  scope='Same-module original output32 and output64; immutable original output32 weights; original F32 mul/add column order')
    (D/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path,str(path.relative_to(ROOT)))
    print(json.dumps(dict(wasm_sha256=result['wasm_sha256'], wasm_bytes=(D/'diagnostic.wasm').stat().st_size)))


if __name__ == '__main__':
    main()
