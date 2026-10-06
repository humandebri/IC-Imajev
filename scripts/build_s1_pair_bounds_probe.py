#!/usr/bin/env python3
"""Build same-module output128 controls and exact pair-bound specialization."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/s1-pair-bounds-v1/build'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specialize(body, keep):
    guard = '(if (local.get $keep) (then'
    found = 0
    while guard in body:
        start = body.index(guard)
        depth = 0
        for end in range(start, len(body)):
            depth += (body[end] == '(') - (body[end] == ')')
            if depth == 0:
                break
        assert body[end-1:end+1] == '))'
        content = body[start+len(guard):end-1]
        body = body[:start] + (content if keep else '') + body[end+1:]
        found += 1
    assert found == 33
    assignment = '(local.set $keep (i32.lt_u (i32.add (local.get $t) (i32.const 1)) (local.get $n)))'
    assert body.count(assignment) == 1
    return body.replace(assignment, '')


def main():
    D.mkdir(parents=True, exist_ok=False)
    original = ROOT/'artifacts/single-quad/build-v2/kernel3.wat'
    odd = ROOT/'artifacts/remaining-efficiency/odd-build-v2/kernel.wat'
    production = json.loads((ROOT/'artifacts/query-packing-v3/build/patch3.json').read_text())
    assert sha(original) == production['source_sha256']
    source = odd.read_text()
    pair_anchor = '(br_if $pairs_done (i32.ge_u (local.get $t) (local.get $fulln)))\n'
    begin = source.index(pair_anchor)+len(pair_anchor)
    end = source.index('(local.set $t (i32.add (local.get $t) (i32.const 2)))', begin)
    pairs = specialize(source[begin:end], True)
    tail_anchor = '(if (i32.lt_u (local.get $t) (local.get $n)) (then\n'
    tail_begin = source.index(tail_anchor, end)+len(tail_anchor)
    assert source.endswith('))\n)\n))\n')
    tail_end = len(source)-len('))\n)\n))\n')
    tail = specialize(source[tail_begin:tail_end], False)
    for m in [1,3,5]:
        initial = f'(local.set $p{m} (v128.const i32x4 0 0 0 0))\n'
        assert tail.count(initial) == 1
        tail = tail.replace(initial, '')
    # Only the first output row is retained: C0=p0+p3-p4+p6,
    # C1=p2+p4. For zero-padded token1, p3=0. This is integer
    # addition before conversion; no F32 reassociation is performed.
    fragment = 'local.get $p0\nlocal.get $p3\ni32x4.add\n'
    assert tail.count(fragment) == 32
    tail = tail.replace(fragment, 'local.get $p0\n')
    assert not re.search(r'local.get \$p[135]\b', tail)
    assert '$sx1' not in tail and '$keep' not in tail
    candidate = source[:begin]+pairs+source[end:tail_begin]+tail+source[tail_end:]
    candidate = candidate.replace('__imajev_s1_wide_accumulate','__imajev_s1_bounded_accumulate')
    (D/'kernel.wat').write_text(candidate)
    (D/'generator.json').write_text(json.dumps(dict(
        original_sha256=sha(original),odd_sha256=sha(odd),candidate_sha256=sha(D/'kernel.wat'),
        complete_pair_guards_removed=33,tail_false_guards_removed=33,
        tail_zero_additions_removed=32,first_pair_unchanged=True,
        complete_pair_bound='t < n & -2 after unchanged first pair',
        tail_bound='t < n after complete-pair loop implies last odd token',
        scope=__doc__),indent=2)+'\n')
    (D/'src').mkdir()
    old_src = ROOT/'scripts/s1_wide_bench/src'
    lib = (old_src/'lib.rs').read_text().replace('assert!(method<=3)', 'assert!(method==3||method==4)')
    lib = lib.replace('if method==3 {', 'if method>=3 {')
    lib = lib.replace('project_wide(&q,prepared.as_ref().unwrap(),&f.scales[..rows],rows)',
                      'project_wide(&q,prepared.as_ref().unwrap(),&f.scales[..rows],rows,method==4)')
    (D/'src/lib.rs').write_text(lib)
    exact = (old_src/'exact.rs').read_text()
    exact = exact.replace('pub fn project_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)',
                          'pub fn project_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize,candidate:bool)')
    exact = exact.replace('self.simd_wide(q,a,sw,rows)', 'self.simd_wide(q,a,sw,rows,candidate)')
    exact = exact.replace('unsafe fn simd_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)',
                          'unsafe fn simd_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize,candidate:bool)')
    exact = exact.replace('for b in 0..cols/256{crate::kernel::wide(',
                          'let kernel=if candidate{crate::kernel::bounded}else{crate::kernel::wide};for b in 0..cols/256{kernel(')
    (D/'src/exact.rs').write_text(exact)
    kernel = (old_src/'kernel.rs').read_text()
    second = kernel[kernel.index('#[export_name="__imajev_s1_wide_accumulate"]'):]
    second = second.replace('__imajev_s1_wide_accumulate','__imajev_s1_bounded_accumulate').replace('fn wide(', 'fn bounded(')
    (D/'src/kernel.rs').write_text(kernel+'\n'+second)
    old = json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text())
    command = old['command'][:]
    command[command.index('--edition=2021')+1] = str(D/'src/lib.rs')
    command[command.index('-o')+1] = str(D/'raw.wasm')
    identities = {str(p.relative_to(ROOT)):sha(p) for p in list((D/'src').glob('*.rs')) +
                  [original,odd,D/'kernel.wat',Path(__file__)]}
    for p, digest in old['dependency_hashes'].items():
        assert sha(ROOT/p) == digest
    with (D/'compiler.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),check=True,stdout=log,stderr=log)
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'
    patches = []
    for previous,wat,out,symbol in [(D/'raw.wasm',original,D/'control.wasm','__imajev_s1_wide_accumulate'),
                                    (D/'control.wasm',D/'kernel.wat',D/'diagnostic.wasm','__imajev_s1_bounded_accumulate')]:
        patches.append(json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True)))
    assert identities == {p:sha(ROOT/p) for p in identities}
    for p,digest in old['dependency_hashes'].items():
        assert sha(ROOT/p) == digest
    report = dict(command=command,explicit_env=old['explicit_env'],source_hashes=identities,
                  dependency_hashes=old['dependency_hashes'],patches=patches,
                  wasm_sha256=sha(D/'diagnostic.wasm'),patcher_sha256=sha(patcher),scope=__doc__)
    (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in identities:
            archive.write(ROOT/p,p)
    print(json.dumps(dict(wasm_sha256=report['wasm_sha256'],wasm_bytes=(D/'diagnostic.wasm').stat().st_size)))


if __name__ == '__main__':
    main()
