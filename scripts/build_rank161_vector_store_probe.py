#!/usr/bin/env python3
"""Vectorize rank161 output reduction/scales/stores, retaining integer DAG."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile
from build_rank161_prepared_probe import reconstruction

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = ROOT/'artifacts/rank161-prepared-v1'
    prior = json.loads((old/'build/report.json').read_text())
    assert json.loads((old/'post-report-audit.json').read_text())['complete']
    for k in ('source_hashes','dependency_hashes'):
        assert all(sha(ROOT/p)==h for p,h in prior[k].items())
    d = ROOT/'artifacts/rank161-vector-store-v1/build'
    d.mkdir(parents=True,exist_ok=False)
    wat = (old/'build/kernel.wat').read_text()
    original = wat
    roots = reconstruction(json.loads((ROOT/'artifacts/rank23-symbolic-v1/mixed-2-3.json').read_text()))[1]
    modifications = []
    def replace(before, after):
        nonlocal wat
        assert wat.count(before) == 1
        wat = wat.replace(before, after)
        modifications.append((before, after))
    for ti in range(6):
        replace(f'(local $sx{ti} f32)',f'(local $sx{ti} v128)')
        before = f'(local.set $sx{ti} (f32.load '
        after = f'(local.set $sx{ti} (v128.load32_splat '
        replace(before,after)
    for ri in range(48):
        replace(f'(local $sw{ri} f32)\n','')
        replace(f'(local.set $sw{ri} (f32.load offset={ri*4} (local.get $sw)))\n','')
    anchor = '(local.set $w0_0_0 '
    index = wat.index(anchor)
    declarations = '\n'.join([f'(local $swv{i} v128)' for i in range(12)]+[f'(local $h{i} v128)' for i in range(6)]+[f'(local $g{i} v128)' for i in range(3)])+'\n'
    loads = '\n'.join(f'(local.set $swv{i} (v128.load offset={i*16} (local.get $sw)))' for i in range(12))+'\n'
    # The removed scale initializers leave only declarations before this point.
    wat = wat[:index]+declarations+loads+wat[index:]
    insertion = declarations+loads
    def shuffle(x,y,indices):
        return [f'local.get ${x}',f'local.get ${y}','i8x16.shuffle '+' '.join(map(str,indices))]
    cursor = 0
    for j in range(4):
        for ti in range(6):
            header = f'(if (i32.lt_u (i32.add (local.get $t) (i32.const {ti})) (local.get $n)) (then\n(local.set $yp '
            start = wat.index(header,cursor)
            end = wat.index('\n))',start)+3
            before = wat[start:end]
            assert before.count('f32.store offset=') == 12
            yp_end = before.index('\n',before.index('(local.set $yp '))
            lines = [before[:yp_end]]
            for ci in range(6):
                pc = f'pc{roots[ti][ci]}'
                lines += [f'local.get ${pc}']+shuffle(pc,pc,[*range(4,8),*range(0,4),*range(12,16),*range(8,12)])
                lines += ['i32x4.add',f'local.set $h{ci}']
            for group in range(3):
                lines += shuffle(f'h{group*2}',f'h{group*2+1}',[*range(0,4),*range(16,20),*range(8,12),*range(24,28)])
                lines.append(f'local.set $g{group}')
            gathers = [('g0','g1',list(range(8))+list(range(16,24))),
                       ('g2','g0',list(range(8))+list(range(24,32))),
                       ('g1','g2',list(range(8,16))+list(range(24,32)))]
            for group,(x,y,indices) in enumerate(gathers):
                offset = j*48+group*16
                lines += ['local.get $yp','local.get $yp',f'v128.load offset={offset}']
                lines += shuffle(x,y,indices)
                lines += ['f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',
                          f'local.get $swv{j*3+group}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
            lines.append('))')
            after = '\n'.join(lines)
            # Repeated token headers occur once for each j; replace the first remaining scalar one.
            wat = wat[:start]+after+wat[end:]
            cursor = start+len(after)
            modifications.append((before,after))
    # Reverse all changed disjoint snippets. Empty deletions are restored by
    # separately removing declarations/loads from the original for comparison.
    reduced_original = original
    for before,after in modifications:
        if after == '':
            reduced_original = reduced_original.replace(before,'')
    reversed_wat = wat.replace(insertion,'',1)
    for before,after in reversed(modifications):
        if after:
            assert reversed_wat.count(after) == 1
            reversed_wat = reversed_wat.replace(after,before,1)
    assert reversed_wat == reduced_original
    assert 'f32.store ' not in wat
    assert wat.count('v128.store offset=') == 6*4*3
    assert wat.count('i32x4.dot_i16x8_s') == original.count('i32x4.dot_i16x8_s')
    (d/'kernel.wat').write_text(wat)
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    patch = json.loads(subprocess.check_output([str(patcher),str(old/'build/diagnostic.wasm'),str(d/'kernel.wat'),str(d/'diagnostic.wasm'),'__imajev_s3_stream_accumulate'],text=True))
    assert patch['wasmparser_validation']
    paths = [Path(__file__),ROOT/'scripts/build_rank161_prepared_probe.py',old/'build/report.json',old/'build/kernel.wat',old/'post-report-audit.json',d/'kernel.wat']
    hashes = dict(prior['source_hashes'])
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    report = dict(prior,wasm_sha256=sha(d/'diagnostic.wasm'),source_hashes=hashes,
                  patches=[patch],locals=len(re.findall(r'\(local \$',wat)),
                  reverse_integer_source_audit=True,output_simd_stores=72,scope=__doc__)
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(d/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in hashes:
            z.write(ROOT/path,path)
    print(json.dumps({'module':report['wasm_sha256'],'locals':report['locals']}))


if __name__ == '__main__':
    main()
