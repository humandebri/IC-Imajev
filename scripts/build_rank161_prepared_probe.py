#!/usr/bin/env python3
"""Diagnostic rank161 SIMD lower-bound probe, with immutable prepared weights.

Sun MIT rank23 data are pinned/audited, then composed with Strassen rank7.
Full coefficient storage is intentionally diagnostic; no production adoption.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/rank161-prepared-v1/build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expr(terms, kind, get):
    code = []
    for i, (name, sign) in enumerate(terms.items()):
        if i == 0 and sign < 0:
            code.append('(v128.const i32x4 0 0 0 0)')
        code.append(get(name))
        if i or sign < 0:
            code.append(f'{kind}.{"add" if sign > 0 else "sub"}')
    return code or ['(v128.const i32x4 0 0 0 0)']


def rust_expr(terms):
    values = list(terms.items())
    name, sign = values[0]
    result = name if sign > 0 else f'i16x8_sub(i16x8_splat(0),{name})'
    for name, sign in values[1:]:
        result = f'i16x8_{"add" if sign > 0 else "sub"}({result},{name})'
    return result


def reconstruction(plan):
    roots = [name for row in plan['outputs'] for name in row]
    uses = Counter(roots)
    for _, terms in plan['nodes']['p']:
        uses.update(terms)
    mapping = {f'p{i}':i for i in range(161)}
    free = [i for i in range(161) if not uses[f'p{i}']]
    capacity = 161
    code = []
    for name, terms in plan['nodes']['p']:
        code += expr(terms, 'i32x4', lambda n:f'local.get $pc{mapping[n]}')
        for parent in terms:
            uses[parent] -= 1
            if uses[parent] == 0:
                free.append(mapping[parent])
        slot = free.pop() if free else capacity
        if slot == capacity:
            capacity += 1
        mapping[name] = slot
        code.append(f'local.set $pc{slot}')
    assert len({mapping[n] for n in roots}) == len(set(roots))
    return code, [[mapping[n] for n in row] for row in plan['outputs']], capacity


def main():
    evidence = ROOT/'artifacts/rank23-symbolic-v1'
    proof = json.loads((evidence/'report.json').read_text())
    native = json.loads((evidence/'native-audit.json').read_text())
    assert proof['complete'] and native['complete'] and native['case_count'] == 80
    for data in (proof, native):
        for p, h in data['hashes'].items():
            assert sha(ROOT/p) == h
    plan = json.loads((evidence/'mixed-2-3.json').read_text())
    assert plan['integer_identity'] and plan['rank'] == 161
    original = ROOT/'artifacts/s3-stream-v1/build'
    old = json.loads((original/'report.json').read_text())
    for key in ('source_hashes','dependency_hashes'):
        assert all(sha(ROOT/p) == h for p,h in old[key].items())
    D.mkdir(parents=True, exist_ok=False)
    src = D/'src'
    src.mkdir()
    for p in (original/'src').glob('*.rs'):
        (src/p.name).write_bytes(p.read_bytes())
    prep = '''use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
pub(super) unsafe fn input(q:&imajev_runtime::int8_kernel::QuantizedRows,block:usize)->Vec<i16>{
let groups=q.rows().div_ceil(6);let mut out=Vec::<i16>::with_capacity(groups*161*88);
for group in 0..groups {for k in 0..11 {
'''
    for i in range(36):
        row, segment = divmod(i,6)
        prep += f'''let a{i}=if group*6+{row}>=q.rows() {{i16x8_splat(0)}} else {{
let pos=(group*6+{row})*q.cols()+block*256+{segment}*43+k*4;
if k<10 {{v128_load64_splat(q.values().as_ptr().add(pos).cast())}} else {{
let tail:[i16;4]=core::array::from_fn(|x|if k*4+x<43 && {segment}*43+k*4+x<256 {{q.values()[pos+x]}} else {{0}});
v128_load64_splat(tail.as_ptr().cast())}} }};
'''
    for name, terms in plan['nodes']['a']:
        prep += f'let {name}={rust_expr(terms)};\n'
    for m,(name,_) in enumerate(plan['leaves']):
        prep += f'v128_store(out.as_mut_ptr().add((group*161+{m})*88+k*8).cast(),{name});\n'
    prep += '}}out.set_len(groups*161*88);out}\n'
    prep += '''#[target_feature(enable="simd128")]
pub(super) unsafe fn weights(w:&[i8],rows:usize,cols:usize)->Vec<i16>{
let tiles=rows.div_ceil(48);let blocks=cols/256;let length=tiles*blocks*4*161*88;
let mut out=Vec::<i16>::with_capacity(length);
for tile in 0..tiles {for block in 0..blocks {for j in 0..4 {for k in 0..11 {
'''
    for i in range(36):
        segment, col = divmod(i,6)
        prep += f'''let bytes:[i16;8]=core::array::from_fn(|x|{{let row=tile*48+j*12+{col}+(x/4)*6;let kk={segment}*43+k*4+x%4;
if row<rows && k*4+x%4<43 && kk<256 {{w[row*cols+block*256+kk]as i16}}else{{0}}}});
let b{i}=v128_load(bytes.as_ptr().cast());
'''
    for name, terms in plan['nodes']['b']:
        prep += f'let {name}={rust_expr(terms)};\n'
    for m,(_,name) in enumerate(plan['leaves']):
        prep += f'v128_store(out.as_mut_ptr().add(((((tile*blocks+block)*4+j)*161+{m})*11+k)*8).cast(),{name});\n'
    prep += '}}}}out.set_len(length);out}\n'
    (src/'prepare_s3.rs').write_text(prep)
    exact = (src/'exact.rs').read_text()
    begin = exact.index('pub fn project_s3(')
    end = exact.index(' pub fn new(',begin)
    project = '''pub fn project_s3(&self,q:&QuantizedRows,sw:&[f32],rows:usize)->Result<Vec<f32>> {
if q.rows()==0||q.rows()>132||q.cols()!=self.cols||rows==0||rows>self.rows||sw.len()!=rows||q.rows()*rows>900_000||!sw.iter().all(|v|v.is_finite()&&*v>0.){return Err("Rank161 bounds".into());}
#[cfg(not(target_arch="wasm32"))]return Err("Wasm diagnostic only".into());
#[cfg(target_arch="wasm32")]unsafe{
let padded=rows.div_ceil(48)*48;let mut out=vec![0f32;q.rows()*padded];let mut scales=vec![0f32;padded];scales[..rows].copy_from_slice(sw);
for block in 0..self.cols/256 {
let operands=crate::prepare_s3::input(q,block);
for r in(0..padded).step_by(48){let wp=self.mixed.as_ptr().add(((r/48)*(self.cols/256)+block)*4*161*88);
crate::s3_kernel::accumulate(operands.as_ptr(),wp.cast(),padded,0,q.scales().as_ptr().add(block),self.cols/256,scales.as_ptr().add(r),out.as_mut_ptr().add(r),q.rows());}}
let mut result=Vec::with_capacity(q.rows()*rows);for t in 0..q.rows(){result.extend_from_slice(&out[t*padded..t*padded+rows]);}
if result.iter().any(|v|!v.is_finite()){return Err("Rank161 nonfinite".into());}Ok(result)}}

'''
    exact = exact[:begin]+project+exact[end:]
    exact = exact.replace('pub struct Prepared {data:Vec<i8>,','pub struct Prepared {data:Vec<i8>,mixed:Vec<i16>,')
    exact = exact.replace('Ok(Self{data,rows,cols})','let mixed=unsafe{crate::prepare_s3::weights(w,rows,cols)};Ok(Self{data,mixed,rows,cols})')
    exact = exact.replace('pub fn bytes(&self)->usize {self.data.len()}','pub fn bytes(&self)->usize {self.data.len()+self.mixed.len()*2}')
    (src/'exact.rs').write_text(exact)
    p = src/'lib.rs'
    p.write_text(p.read_text()+'\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}\n')
    (src/'s3_kernel.rs').write_text('''#[export_name="__imajev_s3_stream_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn accumulate(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize){
let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;
for t in 0..n{for i in 0..48{core::ptr::write_volatile(out.add(t*cols+i),f32::from_bits(marker|0x7fc00000));}}}
''')
    recombine, roots, pc_count = reconstruction(plan)
    lines = ['(module (func (export "__imajev_s3_stream_accumulate") '+
             ' '.join(f'(param ${p} i32)' for p in ['q','w','cols','start','sx','stride','sw','out','n']),
             '(local $t i32) (local $qp i32) (local $yp i32)']
    for i in range(pc_count):
        lines.append(f'(local $pc{i} v128)')
    for m in range(161):
        for k in range(11):
            lines.append(f'(local $x{m}_{k} v128)')
            for j in range(4):
                lines.append(f'(local $w{j}_{m}_{k} v128)')
    for ti in range(6):
        lines.append(f'(local $sx{ti} f32)')
    for ri in range(48):
        lines += [f'(local $sw{ri} f32)']
    for ri in range(48):
        lines.append(f'(local.set $sw{ri} (f32.load offset={ri*4} (local.get $sw)))')
    for j in range(4):
        for m in range(161):
            for k in range(11):
                offset = ((j*161+m)*11+k)*16
                lines.append(f'(local.set $w{j}_{m}_{k} (v128.load offset={offset} (local.get $w)))')
    lines += ['(local.set $t (i32.const 0))','(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))',
              '(local.set $qp (i32.add (local.get $q) (i32.mul (i32.div_u (local.get $t) (i32.const 6)) (i32.const 28336))))']
    for ti in range(6):
        lines.append(f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then (local.set $sx{ti} (f32.load (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $stride)) (i32.const 2)))))))')
    for m in range(161):
        for k in range(11):
            lines.append(f'(local.set $x{m}_{k} (v128.load offset={(m*11+k)*16} (local.get $qp)))')
    for j in range(4):
        for m in range(161):
            for k in range(11):
                lines += [f'local.get $x{m}_{k}',f'local.get $w{j}_{m}_{k}','i32x4.dot_i16x8_s']
                if k:
                    lines.append('i32x4.add')
            lines.append(f'local.set $pc{m}')
        lines += recombine
        for ti in range(6):
            lines += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then',
                      f'(local.set $yp (i32.add (local.get $out) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $cols)) (i32.const 2))))']
            for ci in range(6):
                for half in range(2):
                    row = j*12+ci+half*6
                    pc = roots[ti][ci]
                    lines += ['local.get $yp','local.get $yp',f'f32.load offset={row*4}',
                              f'local.get $pc{pc}',f'i32x4.extract_lane {half*2}',
                              f'local.get $pc{pc}',f'i32x4.extract_lane {half*2+1}',
                              'i32.add','f32.convert_i32_s',f'local.get $sx{ti}','f32.mul',
                              f'local.get $sw{row}','f32.mul','f32.add',f'f32.store offset={row*4}']
            lines.append('))')
    lines += ['(local.set $t (i32.add (local.get $t) (i32.const 6)))','br $tokens','))','))']
    wat = '\n'.join(lines)+'\n'
    (D/'kernel.wat').write_text(wat)
    locals_count = len(re.findall(r'\(local \$',wat))
    assert locals_count <= 10000
    command = old['command'][:]
    command[command.index('--edition=2021')+1] = str(src/'lib.rs')
    command[command.index('-o')+1] = str(D/'raw.wasm')
    with (D/'compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),stdout=log,stderr=log,check=True)
    control = ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    patches = []
    previous = D/'raw.wasm'
    for file,symbol,output in ((control,'__imajev_s1_wide_accumulate',D/'control.wasm'),
                               (D/'kernel.wat','__imajev_s3_stream_accumulate',D/'diagnostic.wasm')):
        row = json.loads(subprocess.check_output([str(patcher),str(previous),str(file),str(output),symbol],text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        previous = output
    paths = [Path(__file__),original/'report.json',control,D/'kernel.wat',evidence/'report.json',evidence/'native-audit.json',evidence/'mixed-2-3.json',evidence/'upstream/LICENSE']+list(src.glob('*.rs'))
    result = dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
                  dependency_hashes=old['dependency_hashes'],command=command,explicit_env=old['explicit_env'],
                  patches=patches,patcher_sha256=sha(patcher),rank=161,token_group=6,output_tile=48,
                  locals=locals_count,reconstruction_registers=pc_count,diagnostic_only=True,
                  immutable_weight_preparation_excluded=True,scope=__doc__)
    (D/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({'module':result['wasm_sha256'],'locals':locals_count}))


if __name__ == '__main__':
    main()
