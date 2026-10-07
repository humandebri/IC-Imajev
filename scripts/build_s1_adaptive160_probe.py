#!/usr/bin/env python3
"""Retest exact output160 with output128/32 tails instead of padding them to128."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/s1-adaptive160-v1/build'


def kernel32():
    wat = (ROOT / 'artifacts/s1-pair-bounds-v1/build/full-kernel.wat').read_text()
    row = '(local.set $row0 (i32.add (local.get $wp0) (i32.mul (local.get $cols) (i32.const 8))))'
    start = wat.index(row)
    end = wat.index('(local.set $t (i32.const 2))', start)
    wat = wat[:start] + wat[end:]
    anchor = 'local.get $x0_0\nlocal.get $w0_8_0'
    start = wat.index(anchor)
    end = wat.index('(local.set $t (i32.add (local.get $t) (i32.const 2)))', start)
    wat = wat[:start] + wat[end:]
    start = wat.index(anchor)
    footer = '))\n)\n))\n'
    assert wat.endswith(footer)
    wat = wat[:start] + footer
    def declaration(m):
        return '' if int(m[1]) >= 8 else m[0]
    wat = re.sub(r'\(local \$(?:w[0-6]|b12|b21)_(\d+)_\d+ v128\)', declaration, wat)
    wat = re.sub(r'\(local \$sw(\d+) v128\)', declaration, wat)
    wat = re.sub(r'\(local.set \$sw(\d+) \(v128.load offset=\d+ \(local.get \$sw\)\)\)', declaration, wat)
    wat = re.sub(r'(?m)^(v128\.(?:load|store) offset=)(\d+)$', lambda m:m[1]+str(int(m[2])-384 if int(m[2])>=512 else int(m[2])), wat)
    wat = wat.replace('(i32.shl (local.get $t) (i32.const 9))', '(i32.shl (local.get $t) (i32.const 7))')
    wat = wat.replace('__imajev_s1_wide_accumulate', '__imajev_s1_32_accumulate')
    assert wat.count('(') == wat.count(')')
    assert not re.search(r'\$(?:w[0-6]|b12|b21)_(?:[89]|[12]\d|3[01])_', wat)
    return wat


def main():
    D.mkdir(parents=True, exist_ok=False)
    old = ROOT / 'artifacts/s1-output160-v1/build'
    manifest = json.loads((old / 'report.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p,h in manifest['source_hashes'].items())
    for name in ['kernel.wat','generator.json']:
        (D / name).write_bytes((old / name).read_bytes())
    (D / 'kernel32.wat').write_text(kernel32())
    p = ROOT / 'scripts/build_s1_output160.py'
    source = p.read_text().replace('artifacts/s1-output160-v1/build', 'artifacts/s1-adaptive160-v1/build')
    before = 'let width=(rows-r).min(160);let tile=if width==160{160}else{128};'
    assert source.count(before) == 1
    source = source.replace(before, 'let tile=if rows-r>=160{160}else if rows-r>=128{128}else{32};let width=(rows-r).min(tile);')
    source = source.replace('else{crate::kernel::wide(ap.as_ptr()', 'else if tile==128{crate::kernel::wide(ap.as_ptr()')
    before = 'sums.as_mut_ptr(),q.rows());}}'
    assert source.count(before) == 1
    source = source.replace(before, 'sums.as_mut_ptr(),q.rows());}else{crate::kernel::wide32(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,b*256,q.scales().as_ptr().add(b),cols/256,swp,sums.as_mut_ptr(),q.rows());}}')
    before = ";(D/'src/kernel.rs').write_text(kernel)"
    assert source.count(before) == 1
    extra = ''';kernel+='\n#[export_name="__imajev_s1_32_accumulate"] #[inline(never)]\npub(crate) unsafe extern "C" fn wide32(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;for i in 0..n*32{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^632)|0x7fc00000));}}\n';(D/'src/kernel.rs').write_text(kernel)'''
    source = source.replace(before, extra.replace('\n', '\\n'))
    source = source.replace("[D/'kernel.wat',D/'generator.json',control", "[D/'kernel.wat',D/'kernel32.wat',D/'generator.json',control")
    before = "(D/'control.wasm',D/'kernel.wat',D/'diagnostic.wasm','__imajev_s1_160_accumulate')"
    assert source.count(before) == 1
    source = source.replace(before, "(D/'control.wasm',D/'kernel.wat',D/'wide160.wasm','__imajev_s1_160_accumulate'),(D/'wide160.wasm',D/'kernel32.wat',D/'diagnostic.wasm','__imajev_s1_32_accumulate')")
    source = source.replace("patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'", "patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'")
    (D.parent / 'frozen-builder.py').write_text(source)
    (D.parent / 'upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
