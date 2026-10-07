#!/usr/bin/env python3
"""Exact rank49 K256 with two output quartets per vector and late pair reduction."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/s2-pair-late-v1/build-v3'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    D.mkdir(parents=True, exist_ok=False)
    src = D/'src'
    src.mkdir()
    symbolic = (ROOT/'scripts/generate_strassen2.py').read_text()
    g = {'__file__':str(ROOT/'scripts/generate_strassen2.py')}
    exec(compile(symbolic[:symbolic.index("coeff = '//")], g['__file__'], 'exec'), g)
    leaves, result = g['leaves'], g['result']
    # The imported symbolic proof includes the rank49 identity and I32 bounds.
    def combine(e, name, kind):
        out = []
        for i, (m, sign) in enumerate(sorted(e.items())):
            if i == 0 and sign < 0:
                out.append('(v128.const i32x4 0 0 0 0)')
            out.append('local.get $'+name(m))
            if i or sign < 0:
                out.append(kind+'.'+('add' if sign > 0 else 'sub'))
        return out
    prep = '''use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
pub(super) unsafe fn inputs(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,out:*mut i16){
let cols=q.cols();
for group in 0..groups {for block in 0..cols/256 {for k in(0..64).step_by(8){
'''
    for i in range(16):
        prep += f'let x{i}=v128_load(q.values().as_ptr().add((group*4+{i//4})*cols+block*256+{i%4*64}+k).cast());\n'
    for m, (a, _) in enumerate(leaves):
        terms = sorted(a.items())
        value = f'x{terms[0][0]}' if terms[0][1] > 0 else f'i16x8_sub(i16x8_splat(0),x{terms[0][0]})'
        for i, sign in terms[1:]:
            value = f'i16x8_{"add" if sign > 0 else "sub"}({value},x{i})'
        prep += f'let value={value};let dst=out.add(({m}*groups+group)*(cols/2)+block*128+k*2);\n'
        for half in range(2):
            idx = ','.join(str(half*8+i) for _ in range(2) for i in range(8))
            prep += f'v128_store(dst.add({half*8}).cast(),i8x16_shuffle::<{idx}>(value,value));\n'
    prep += '}}}}\n'
    (src/'prepare.rs').write_text(prep)
    shutil.copyfile(ROOT/'scripts/wat_s2_lane_bench/src/coeff.rs', src/'coeff.rs')
    code = (ROOT/'scripts/wat_s2_lane_bench/src/s2.rs').read_text()
    lo = code.index('  for group in 0..rows/16')
    hi = code.index('  Ok(Self', lo)
    code = code[:lo]+'''  for group in 0..rows/8{for block in 0..cols/256{for k in 0..64{for j in 0..16{for lane in 0..2{data[j*plane+group*(cols/2)+block*128+(k/4)*8+lane*4+k%4]=w[(group*8+lane*4+j%4)*cols+block*256+j/4*64+k];}}}}}
'''+code[hi:]
    code = code.replace('m*a.groups*self.cols)', 'm*a.groups*(self.cols/2))')
    code = code.replace('step_by(32)', 'step_by(64)').replace('vec![[0f32;32]', 'vec![[0f32;64]')
    code = code.replace('r..t*rows+r+32', 'r..t*rows+r+64')
    code = code.replace('let len=49*groups*cols;', 'let len=49*groups*(cols/2);')
    # Native fallback uses the new layout too, independently of the WAT body.
    code = code.replace('(row/4)*self.cols+block*256+(k/2)*8+(row%4)*2+k%2', '(row/2)*(self.cols/2)+block*128+(k/4)*8+(row%2)*4+k%4')
    code = code.replace('(m*a.groups+group)*self.cols+block*256+(k/2)*8+k%2', '(m*a.groups+group)*(self.cols/2)+block*128+(k/4)*8+k%4')
    code = code.replace('for lane in 0..4{data[(m*groups+group)*cols+block*256+(k/2)*8+lane*2+k%2]', 'for lane in 0..2{data[(m*groups+group)*(cols/2)+block*128+(k/4)*8+lane*4+k%4]')
    code = code.replace('rows%32!=0||rows>self.rows', 'rows%64!=0||rows>self.rows')
    (src/'s2.rs').write_text(code)
    bench = ROOT/'scripts/s1_wide_bench/src'
    lib = (bench/'lib.rs').read_text()
    lib = lib.replace('pub mod exact;', 'pub mod exact;\npub mod s2;\n#[cfg(target_arch="wasm32")]mod s2_kernel;')
    lib = lib.replace('sealed:bool}', 's2:Option<s2::Prepared>,sealed:bool}').replace('sealed:false}', 's2:None,sealed:false}')
    lib = lib.replace('f.sealed=true;', 'f.s2=Some(s2::Prepared::new(&f.w,f.rows,f.cols).unwrap());f.sealed=true;')
    lib = lib.replace('assert!(method<=3)', 'assert!(method==3||method==4)')
    lib = lib.replace('(method>0).then(', '(method==3).then(')
    lib = lib.replace('let prep=ic_cdk::api::performance_counter(0);', 'let s2_input=(method==4).then(||s2::operands(&q));let prep=ic_cdk::api::performance_counter(0);')
    lib = lib.replace('let out=if method==3 {', 'let out=if method==4 {f.s2.as_ref().unwrap().project(&q,s2_input.as_ref().unwrap(),&f.scales[..rows],rows).unwrap()}else if method==3 {')
    # Include input preparation in the same measured interval as the candidate.
    lib += '\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}\n'
    (src/'lib.rs').write_text(lib)
    for name in ['exact.rs', 'kernel.rs']:
        shutil.copyfile(bench/name, src/name)
    stub = (ROOT/'scripts/wat_s2_lane_bench/src/s2_kernel.rs').read_text().replace('_n*32', '_n*64')
    (src/'s2_kernel.rs').write_text(stub)
    lines = ['(module (func (export "__imajev_s2_raw_accumulate") (param $q i32) (param $w i32) (param $cols i32) (param $start i32) (param $sx i32) (param $stride i32) (param $sw i32) (param $sums i32) (param $n i32)', '(local $t i32) (local $qp i32) (local $yp i32) (local $qoff i32) (local $v v128)']
    for b in range(16):
        lines += [f'(local $wp{b} i32) (local $b{b} v128)']
    for m in range(49):
        lines += [f'(local $p{m} v128)']+[f'(local $x{m}_{k} v128)' for k in range(16)]
        for j in range(8):
            lines += [f'(local $w{m}_{j}_{k} v128)' for k in range(16)]
    for ri in range(4):
        lines += [f'(local $c{ri} v128)']
    lines += ['(local $u0 v128) (local $u1 v128)']
    for ti in range(4):
        lines += [f'(local $sx{ti} v128)']
    for j in range(16):
        lines += [f'(local $sw{j} v128)']
    for j in range(16):
        lines += [f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))']
    for b in range(16):
        lines += [f'(local.set $wp{b} (i32.add (i32.load offset={b*4} (local.get $w)) (i32.shr_u (local.get $start) (i32.const 1))))']
    def row(first):
        out = ['(local.set $qoff (i32.add (i32.shr_u (i32.mul (local.get $t) (local.get $cols)) (i32.const 2)) (local.get $start)))']
        for ti in range(4):
            out += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then (local.set $sx{ti} (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const {ti})) (local.get $stride)) (i32.const 2)))))))']
        for j in range(8):
            if first:
                for k in range(16):
                    for b in range(16):
                        out += [f'(local.set $b{b} (v128.load8x8_s offset={k*8} (i32.add (local.get $wp{b}) (i32.mul (i32.shr_u (local.get $cols) (i32.const 1)) (i32.const {j})))))']
                    for m, (_, b) in enumerate(leaves):
                        out += combine(b, lambda b:f'b{b}', 'i16x8')+[f'local.set $w{m}_{j}_{k}']
            for m in range(49):
                if j == 0:
                    out += [f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (local.get $qoff)))']
                for k in range(16):
                    out += [f'(local.tee $x{m}_{k} (v128.load offset={k*16} (local.get $qp)))' if j == 0 else f'local.get $x{m}_{k}', f'local.get $w{m}_{j}_{k}', 'i32x4.dot_i16x8_s']
                    if k:
                        out += ['i32x4.add']
                out += [f'local.set $p{m}']
            for ti in range(4):
                out += [f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then', f'(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.add (local.get $t) (i32.const {ti})) (i32.const 8))))']
                for ri in range(4):
                    out += combine(result[ti][ri], lambda m:f'p{m}', 'i32x4')+[f'local.set $c{ri}']
                # Pack the first/second K halves, then add once per four outputs.
                for a, b, u in [(0,1,0),(2,3,1)]:
                    out += [f'local.get $c{a}',f'local.get $c{b}', 'i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27',f'local.get $c{a}',f'local.get $c{b}', 'i8x16.shuffle 4 5 6 7 20 21 22 23 12 13 14 15 28 29 30 31', 'i32x4.add', f'local.set $u{u}']
                for half in range(2):
                    idx = list(range(half*8,half*8+8))+list(range(16+half*8,24+half*8))
                    out += ['local.get $yp', f'v128.load offset={j*32+half*16}', 'local.get $u0','local.get $u1','i8x16.shuffle '+' '.join(map(str,idx)), 'f32x4.convert_i32x4_s', f'local.get $sx{ti}', 'f32x4.mul', f'local.get $sw{j*2+half}', 'f32x4.mul','f32x4.add','local.set $v','local.get $yp','local.get $v',f'v128.store offset={j*32+half*16}']
                out += ['))']
        return out
    lines += ['(block $done','(br_if $done (i32.eqz (local.get $n)))']+row(True)+['(local.set $t (i32.const 4))','(loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+row(False)+['(local.set $t (i32.add (local.get $t) (i32.const 4)))','br $tokens','))','))']
    (D/'kernel.wat').write_text('\n'.join(lines)+'\n')
    locals_count = sum(line.count('(local $') for line in lines)
    assert locals_count < 10000
    old = json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text())
    cmd = old['command'][:]
    cmd[cmd.index('--edition=2021')+1] = str(src/'lib.rs')
    cmd[cmd.index('-o')+1] = str(D/'raw.wasm')
    for p,h in old['dependency_hashes'].items():
        assert sha(ROOT/p) == h
    with (D/'compiler.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),stdout=log,stderr=log,check=True)
    previous = D/'raw.wasm'
    patches = []
    control = ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
    for wat, symbol, name in [(control,'__imajev_s1_wide_accumulate','control.wasm'),(D/'kernel.wat','__imajev_s2_raw_accumulate','diagnostic.wasm')]:
        output = D/name
        patch = json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'), str(previous), str(wat), str(output), symbol],text=True))
        assert patch['wasmparser_validation']
        patches.append(patch)
        previous = output
    sources = [Path(__file__),D/'kernel.wat',control,ROOT/'scripts/generate_strassen2.py']+list(src.glob('*.rs'))
    report = dict(wasm_sha256=sha(D/'diagnostic.wasm'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},dependency_hashes=old['dependency_hashes'],command=cmd,explicit_env=old['explicit_env'],patches=patches,locals=locals_count,output_tile=64,symbolic_identity_checked=True,i32_bound_checked=True,scope=__doc__)
    (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(dict(module=report['wasm_sha256'],locals=locals_count)))


if __name__ == '__main__':
    main()
