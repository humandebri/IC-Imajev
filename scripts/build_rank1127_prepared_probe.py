#!/usr/bin/env python3
"""Generate diagnostic rank1127, eight distinct K lanes, full operand caches."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile
from build_rank161_prepared_probe import rust_expr

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/rank1127-prepared-v1/build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    evidence = ROOT/'artifacts/rank23-symbolic-v1'
    audit = json.loads((evidence/'rank1127-simd-layout-audit.json').read_text())
    assert audit['complete'] and audit['case_count']==48
    for p,h in audit['hashes'].items():
        assert sha(ROOT/p)==h
    plan = json.loads((evidence/'mixed-2-3-2.json').read_text())
    old = ROOT/'artifacts/rank161-prepared-v1/build'
    prior = json.loads((old/'report.json').read_text())
    for key in ('source_hashes','dependency_hashes'):
        assert all(sha(ROOT/p)==h for p,h in prior[key].items())
    D.mkdir(parents=True,exist_ok=False)
    src = D/'src'
    src.mkdir()
    for p in (old/'src').glob('*.rs'):
        (src/p.name).write_bytes(p.read_bytes())
    prep = '''use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
pub(super) unsafe fn input(q:&imajev_runtime::int8_kernel::QuantizedRows,block:usize)->Vec<i16>{
let groups=q.rows().div_ceil(12);let len=groups*1127*24;let mut out=Vec::<i16>::with_capacity(len);
for group in 0..groups{for k in 0..3{
'''
    for i in range(144):
        row,segment = divmod(i,12)
        prep += f'''let a{i}=if group*12+{row}>=q.rows(){{i16x8_splat(0)}}else{{
let kk={segment}*22+k*8;let pos=(group*12+{row})*q.cols()+block*256+kk;
if k*8+7<22 && kk+7<256{{v128_load(q.values().as_ptr().add(pos).cast())}}else{{
let tail:[i16;8]=core::array::from_fn(|x|if k*8+x<22&&kk+x<256{{q.values()[pos+x]}}else{{0}});v128_load(tail.as_ptr().cast())}}}};
'''
    for name,terms in plan['nodes']['a']:
        prep += f'let {name}={rust_expr(terms)};\n'
    for m,(name,_) in enumerate(plan['leaves']):
        prep += f'v128_store(out.as_mut_ptr().add((group*1127+{m})*24+k*8).cast(),{name});\n'
    prep += '}}out.set_len(len);out}\n'
    prep += '''#[target_feature(enable="simd128")]
pub(super) unsafe fn weights(w:&[i8],rows:usize,cols:usize)->Vec<i16>{
let tiles=rows.div_ceil(12);let blocks=cols/256;let len=tiles*blocks*1127*24;let mut out=Vec::<i16>::with_capacity(len);
for tile in 0..tiles{for block in 0..blocks{for k in 0..3{
'''
    for i in range(144):
        segment,col = divmod(i,12)
        prep += f'''let row=tile*12+{col};let kk={segment}*22+k*8;
let bytes:[i16;8]=core::array::from_fn(|x|if row<rows&&k*8+x<22&&kk+x<256{{w[row*cols+block*256+kk+x]as i16}}else{{0}});
let b{i}=v128_load(bytes.as_ptr().cast());
'''
    for name,terms in plan['nodes']['b']:
        prep += f'let {name}={rust_expr(terms)};\n'
    for m,(_,name) in enumerate(plan['leaves']):
        prep += f'v128_store(out.as_mut_ptr().add(((tile*blocks+block)*1127+{m})*24+k*8).cast(),{name});\n'
    prep += '}}}out.set_len(len);out}\n'
    (src/'prepare_s3.rs').write_text(prep)
    p = src/'exact.rs'
    exact = p.read_text()
    assert exact.count('rows.div_ceil(48)*48')==1
    exact = exact.replace('rows.div_ceil(48)*48','rows.div_ceil(12)*12').replace('step_by(48)','step_by(12)').replace('((r/48)*(self.cols/256)+block)*4*161*88','((r/12)*(self.cols/256)+block)*1127*24')
    p.write_text(exact)
    p = src/'s3_kernel.rs'
    p.write_text(p.read_text().replace('for i in 0..48','for i in 0..12'))
    nodes = dict(plan['nodes']['p'])
    roots = [n for row in plan['outputs'] for n in row]
    uses = Counter(roots)
    for terms in nodes.values():
        uses.update(terms.keys())
    retained = {n:f'rc{i}' for i,n in enumerate(n for n,_ in plan['nodes']['p'] if uses[n]>1 or n in roots)}
    def expression(n,expand=False):
        if n not in nodes:
            return [f'local.get $pc{n[1:]}']
        if not expand and n in retained:
            return [f'local.get ${retained[n]}']
        code = []
        for i,(parent,sign) in enumerate(nodes[n].items()):
            if i==0 and sign<0:
                code.append('(v128.const i32x4 0 0 0 0)')
            code += expression(parent)
            if i or sign<0:
                code.append('i32x4.add' if sign>0 else 'i32x4.sub')
        return code
    recombine = []
    for n,_ in plan['nodes']['p']:
        if n in retained:
            recombine += expression(n,True)+[f'local.set ${retained[n]}']
    # Independently compare emitted stack expressions to saved C-DAG polynomials.
    values = {f'pc{i}':{i:1} for i in range(1127)}
    stack = []
    for line in recombine:
        if line.startswith('local.get $'):
            stack.append(dict(values[line.split('$')[1]]))
        elif line.startswith('local.set $'):
            values[line.split('$')[1]] = stack.pop()
        elif line.startswith('(v128.const'):
            stack.append({})
        else:
            right,left = stack.pop(),stack.pop()
            sign = 1 if line=='i32x4.add' else -1
            for k,v in right.items():
                left[k] = left.get(k,0)+sign*v
            stack.append({k:v for k,v in left.items() if v})
    reference = {f'p{i}':{i:1} for i in range(1127)}
    for n,terms in nodes.items():
        result = {}
        for parent,sign in terms.items():
            for k,v in reference[parent].items():
                result[k] = result.get(k,0)+sign*v
        reference[n] = {k:v for k,v in result.items() if v}
    assert not stack and all(values[retained[n]]==reference[n] for n in roots)
    lines = ['(module (func (export "__imajev_s3_stream_accumulate") '+
             ' '.join(f'(param ${p} i32)' for p in ['q','w','cols','start','sx','stride','sw','out','n']),
             '(local $t i32) (local $qp i32) (local $yp i32)']
    for m in range(1127):
        lines.append(f'(local $pc{m} v128)')
        for k in range(3):
            lines.append(f'(local $x{m}_{k} v128) (local $w{m}_{k} v128)')
    for n in retained.values():
        lines.append(f'(local ${n} v128)')
    for ti in range(12):
        lines.append(f'(local $sx{ti} v128)')
    for i in range(3):
        lines.append(f'(local $swv{i} v128)')
    for i in range(4):
        lines.append(f'(local $h{i} v128)')
    lines += ['(local $g0 v128) (local $g1 v128)']
    for i in range(3):
        lines.append(f'(local.set $swv{i} (v128.load offset={i*16} (local.get $sw)))')
    for m in range(1127):
        for k in range(3):
            lines.append(f'(local.set $w{m}_{k} (v128.load offset={(m*3+k)*16} (local.get $w)))')
    lines += ['(local.set $t (i32.const 0))','(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))',
              '(local.set $qp (i32.add (local.get $q) (i32.mul (i32.div_u (local.get $t) (i32.const 12)) (i32.const 54096))))']
    for ti in range(12):
        lines.append(f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then (local.set $sx{ti} (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $stride)) (i32.const 2)))))))')
    for m in range(1127):
        for k in range(3):
            lines.append(f'(local.set $x{m}_{k} (v128.load offset={(m*3+k)*16} (local.get $qp)))')
    for m in range(1127):
        for k in range(3):
            lines += [f'local.get $x{m}_{k}',f'local.get $w{m}_{k}','i32x4.dot_i16x8_s']
            if k:
                lines.append('i32x4.add')
        lines.append(f'local.set $pc{m}')
    lines += recombine
    def shuffle(x,y,indices):
        return [f'local.get ${x}',f'local.get ${y}','i8x16.shuffle '+' '.join(map(str,indices))]
    for ti in range(12):
        lines += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then',
                  f'(local.set $yp (i32.add (local.get $out) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $cols)) (i32.const 2))))']
        for group in range(3):
            for ci in range(4):
                rc = retained[plan['outputs'][ti][group*4+ci]]
                lines += [f'local.get ${rc}']+shuffle(rc,rc,[*range(4,8),*range(0,4),*range(12,16),*range(8,12)])
                lines += ['i32x4.add',f'local.set $h{ci}']
                h = f'h{ci}'
                lines += [f'local.get ${h}']+shuffle(h,h,[*range(8,16),*range(0,8)])
                lines += ['i32x4.add',f'local.set ${h}']
            for pair in range(2):
                lines += shuffle(f'h{pair*2}',f'h{pair*2+1}',[*range(0,4),*range(16,20),*range(0,4),*range(16,20)])
                lines.append(f'local.set $g{pair}')
            lines += ['local.get $yp','local.get $yp',f'v128.load offset={group*16}']
            lines += shuffle('g0','g1',[*range(0,8),*range(16,24)])
            lines += ['f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $swv{group}','f32x4.mul','f32x4.add',f'v128.store offset={group*16}']
        lines.append('))')
    lines += ['(local.set $t (i32.add (local.get $t) (i32.const 12)))','br $tokens','))','))']
    wat = '\n'.join(lines)+'\n'
    (D/'kernel.wat').write_text(wat)
    locals_count = len(re.findall(r'\(local \$',wat))
    assert locals_count<10000
    command = prior['command'][:]
    command[command.index('--edition=2021')+1] = str(src/'lib.rs')
    command[command.index('-o')+1] = str(D/'raw.wasm')
    with (D/'compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**prior['explicit_env']),stdout=log,stderr=log,check=True)
    control = ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    previous = D/'raw.wasm'
    patches = []
    for file,symbol,output in ((control,'__imajev_s1_wide_accumulate',D/'control.wasm'),(D/'kernel.wat','__imajev_s3_stream_accumulate',D/'diagnostic.wasm')):
        row = json.loads(subprocess.check_output([str(patcher),str(previous),str(file),str(output),symbol],text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        previous = output
    paths = [Path(__file__),ROOT/'scripts/build_rank161_prepared_probe.py',old/'report.json',control,D/'kernel.wat',evidence/'rank1127-simd-layout-audit.json',evidence/'mixed-2-3-2.json',evidence/'upstream/LICENSE']+list(src.glob('*.rs'))
    result = dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},dependency_hashes=prior['dependency_hashes'],command=command,explicit_env=prior['explicit_env'],patches=patches,patcher_sha256=sha(patcher),rank=1127,token_group=12,output_tile=12,locals=locals_count,reconstruction_retained=len(retained),emitted_integer_stack_identity_verified=True,diagnostic_only=True,immutable_weight_preparation_excluded=True,scope=__doc__)
    (D/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({'module':result['wasm_sha256'],'locals':locals_count}))


if __name__=='__main__':
    main()
