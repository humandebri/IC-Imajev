#!/usr/bin/env python3
"""Inline single-use integer reconstruction expressions of rank161 SIMD."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from build_rank161_prepared_probe import reconstruction

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = ROOT/'artifacts/rank161-vector-store-v1'
    prior = json.loads((old/'build/report.json').read_text())
    assert json.loads((old/'post-report-audit.json').read_text())['complete']
    for key in ('source_hashes','dependency_hashes'):
        assert all(sha(ROOT/p)==h for p,h in prior[key].items())
    plan = json.loads((ROOT/'artifacts/rank23-symbolic-v1/mixed-2-3.json').read_text())
    nodes = dict(plan['nodes']['p'])
    roots = [n for row in plan['outputs'] for n in row]
    uses = Counter(roots)
    for terms in nodes.values():
        uses.update(terms.keys())
    retain = {n:f'rc{i}' for i,n in enumerate(n for n,_ in plan['nodes']['p'] if uses[n]>1 or n in roots)}
    def expression(name, expand=False):
        if name not in nodes:
            return [f'local.get $pc{int(name[1:])}']
        if not expand and name in retain:
            return [f'local.get ${retain[name]}']
        code = []
        for i,(parent,sign) in enumerate(nodes[name].items()):
            if i==0 and sign<0:
                code.append('(v128.const i32x4 0 0 0 0)')
            code += expression(parent)
            if i or sign<0:
                code.append('i32x4.add' if sign>0 else 'i32x4.sub')
        return code
    code = []
    for name,_ in plan['nodes']['p']:
        if name in retain:
            code += expression(name,True)+[f'local.set ${retain[name]}']
    # Independent symbolic stack interpreter of the emitted WAT instructions.
    values = {f'pc{i}':{i:1} for i in range(161)}
    stack = []
    for line in code:
        if line.startswith('local.get $'):
            stack.append(dict(values[line.split('$')[1]]))
        elif line.startswith('local.set $'):
            values[line.split('$')[1]] = stack.pop()
        elif line.startswith('(v128.const'):
            stack.append({})
        else:
            assert line in ('i32x4.add','i32x4.sub')
            right,left = stack.pop(),stack.pop()
            sign = 1 if line=='i32x4.add' else -1
            for key,value in right.items():
                left[key] = left.get(key,0)+sign*value
            stack.append({k:v for k,v in left.items() if v})
    assert not stack
    reference = {f'p{i}':{i:1} for i in range(161)}
    for name,terms in plan['nodes']['p']:
        result = {}
        for parent,sign in terms.items():
            for key,value in reference[parent].items():
                result[key] = result.get(key,0)+sign*value
        reference[name] = {k:v for k,v in result.items() if v}
    assert all(values[retain[root]]==reference[root] for root in roots)
    oldcode,oldroots,_ = reconstruction(plan)
    oldbody = '\n'.join(oldcode)
    wat = (old/'build/kernel.wat').read_text()
    parts = wat.split(oldbody)
    assert len(parts)==5
    for j in range(1,5):
        stop = parts[j].find('local.get $x0_0') if j<4 else parts[j].find('(local.set $t ')
        assert stop>0
        fp = parts[j][:stop]
        mapping = {oldroots[i][k]:retain[plan['outputs'][i][k]] for i in range(6) for k in range(6)}
        assert len(mapping)==36
        fp = re.sub(r'local.get \$pc(\d+)',lambda m:'local.get $'+mapping[int(m[1])],fp)
        parts[j] = fp+parts[j][stop:]
    wat = '\n'.join(code).join(parts)
    anchor = '(local.set $swv0 '
    index = wat.index(anchor)
    wat = wat[:index]+'\n'.join(f'(local ${name} v128)' for name in retain.values())+'\n'+wat[index:]
    assert wat.count('i32x4.dot_i16x8_s')==7084
    assert wat.count('v128.store offset=')==72
    assert len(re.findall(r'\(local \$',wat))<=10000
    d = ROOT/'artifacts/rank161-inline-reconstruction-v1/build'
    d.mkdir(parents=True,exist_ok=False)
    (d/'kernel.wat').write_text(wat)
    (d/'reconstruction.wat-fragment').write_text('\n'.join(code)+'\n')
    patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'
    patch = json.loads(subprocess.check_output([str(patcher),str(old/'build/diagnostic.wasm'),str(d/'kernel.wat'),str(d/'diagnostic.wasm'),'__imajev_s3_stream_accumulate'],text=True))
    assert patch['wasmparser_validation']
    paths = [Path(__file__),old/'build/report.json',old/'build/kernel.wat',old/'post-report-audit.json',d/'kernel.wat',d/'reconstruction.wat-fragment']
    hashes = dict(prior['source_hashes'])
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    report = dict(prior,wasm_sha256=sha(d/'diagnostic.wasm'),source_hashes=hashes,patches=[patch],
                  locals=len(re.findall(r'\(local \$',wat)),reconstruction_single_use_inlined=180,
                  reconstruction_retained=len(retain),emitted_integer_stack_identity_verified=True,
                  scope=__doc__)
    (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(d/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in hashes:
            z.write(ROOT/p,p)
    print(json.dumps({'module':report['wasm_sha256'],'locals':report['locals'],'retained':len(retain)}))


if __name__ == '__main__':
    main()
