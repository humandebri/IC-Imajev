#!/usr/bin/env python3
"""Move exact rank343 weight transforms into immutable diagnostic preparation."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile
from generate_s3_stream_probe import plan

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/s3-prepared-v1/build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    D.mkdir(parents=True, exist_ok=False)
    src = D / 'src'
    src.mkdir()
    original = ROOT / 'artifacts/s3-stream-v1/build'
    old = json.loads((original / 'report.json').read_text())
    assert all(sha(ROOT / p) == h for key in ['source_hashes', 'dependency_hashes'] for p, h in old[key].items())
    for p in (original / 'src').glob('*.rs'):
        (src / p.name).write_bytes(p.read_bytes())
    a, b, c, leaves, roots, bound = plan()
    prep = '''use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
pub(super) unsafe fn weights(w:&[i8],rows:usize,cols:usize)->Vec<i16>{
let length=(rows/32)*(cols/256)*2*343*8*8;
let mut out=Vec::<i16>::with_capacity(length);
for tile in 0..rows/32 {for block in 0..cols/256 {for j in 0..2 {for k in 0..8 {
'''
    for ci in range(8):
        for ri in range(8):
            index = ci * 8 + ri
            ptr = f'w.as_ptr().add((tile*32+j*16+{ri})*cols+block*256+{ci}*32+k*4)'
            ptr2 = f'w.as_ptr().add((tile*32+j*16+{ri}+8)*cols+block*256+{ci}*32+k*4)'
            prep += f'let b{index}=i16x8_extend_low_i8x16(i8x16_shuffle::<0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27>(v128_load32_zero({ptr}.cast()),v128_load32_zero({ptr2}.cast())));\n'
    for name, terms in b.nodes:
        values = list(terms.items())
        name0, sign = values[0]
        expr = name0 if sign > 0 else f'i16x8_sub(i16x8_splat(0),{name0})'
        for operand, sign in values[1:]:
            expr = f'i16x8_{"add" if sign > 0 else "sub"}({expr},{operand})'
        prep += f'let {name}={expr};\n'
    for m, (_, name) in enumerate(leaves):
        prep += f'v128_store(out.as_mut_ptr().add(((((tile*(cols/256)+block)*2+j)*343+{m})*8+k)*8).cast(),{name});\n'
    prep += '}}}}out.set_len(length);out}\n'
    (src / 'prepare_weights.rs').write_text(prep)
    p = src / 'exact.rs'
    exact = p.read_text().replace('pub struct Prepared {data:Vec<i8>,', 'pub struct Prepared {data:Vec<i8>,s3:Vec<i16>,')
    exact = exact.replace('Ok(Self{data,rows,cols})', 'let s3=unsafe{crate::prepare_weights::weights(w,rows,cols)};Ok(Self{data,s3,rows,cols})')
    exact = exact.replace('pub fn bytes(&self)->usize {self.data.len()}', 'pub fn bytes(&self)->usize {self.data.len()+self.s3.len()*2}')
    before = '''   let wp:[*const i8;4]=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*cols+(r/4)*cols));
   crate::s3_kernel::accumulate(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,sw.as_ptr().add(r),out.as_mut_ptr().add(r),n);'''
    assert exact.count(before) == 1
    exact = exact.replace(before, '''   let wp=self.s3.as_ptr().add(((r/32)*(cols/256)+block)*2*343*8*8);
   crate::s3_kernel::accumulate(ap.as_ptr().cast(),wp.cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,sw.as_ptr().add(r),out.as_mut_ptr().add(r),n);''')
    p.write_text(exact)
    p = src / 'lib.rs'
    p.write_text(p.read_text() + '\nmod prepare_weights;\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:Principal,rows:u32,cols:u32){init(owner,rows,cols)}\n')
    wat = (original / 'kernel.wat').read_text()
    begin = wat.index('(local.set $bp0 ')
    end = wat.index('(local.set $t (i32.const 0))', begin)
    loads = '\n'.join(f'(local.set $w{j}_{m}_{k} (v128.load offset={((j*343+m)*8+k)*16} (local.get $w)))' for j in range(2) for m in range(343) for k in range(8)) + '\n'
    wat = wat[:begin] + loads + wat[end:]
    # Old transform temporaries are unused after the new fixed loads.
    for i in range(64):
        wat = wat.replace(f'(local $bp{i} i32) (local $b{i} v128)', '')
    for name, _ in b.nodes:
        wat = wat.replace(f'(local ${name} v128)', '')
    assert '$bp0' not in wat
    (D / 'kernel.wat').write_text(wat)
    command = old['command'][:]
    command[command.index('--edition=2021') + 1] = str(src / 'lib.rs')
    command[command.index('-o') + 1] = str(D / 'raw.wasm')
    with (D / 'compiler.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, **old['explicit_env']), stdout=log, stderr=log, check=True)
    control = ROOT / 'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
    patcher = ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    previous = D / 'raw.wasm'
    patches = []
    for file, symbol, output in [(control, '__imajev_s1_wide_accumulate', D / 'control.wasm'), (D / 'kernel.wat', '__imajev_s3_stream_accumulate', D / 'diagnostic.wasm')]:
        row = json.loads(subprocess.check_output([str(patcher), str(previous), str(file), str(output), symbol], text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        previous = output
    paths = [Path(__file__), ROOT / 'scripts/generate_s3_stream_probe.py', original / 'report.json', control, D / 'kernel.wat'] + list(src.glob('*.rs'))
    result = dict(wasm_sha256=sha(previous), source_hashes={str(p.relative_to(ROOT)): sha(p) for p in paths}, dependency_hashes=old['dependency_hashes'], command=command, explicit_env=old['explicit_env'], patches=patches, patcher_sha256=sha(patcher), rank=343, weight_bytes_ratio=343/32, locals=len(re.findall(r'\(local \$', wat)), scope=__doc__)
    (D / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    with zipfile.ZipFile(D / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, str(p.relative_to(ROOT)))
    print(json.dumps(dict(module=result['wasm_sha256'], locals=result['locals'])))


if __name__ == '__main__':
    main()
