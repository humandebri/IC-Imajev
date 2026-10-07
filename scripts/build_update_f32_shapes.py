#!/usr/bin/env python3
"""Combine adaptive INT8 with output512 B and columns512 A stack kernels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    upstream = ROOT / 'artifacts/update-adaptive160-v2'
    hashes = json.loads((upstream/'workflow-hashes.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    for name in ['f32-stack512-v1', 'f32-columns512-v1']:
        proof = json.loads((ROOT/'artifacts'/name/'summary.json').read_text())
        assert proof['native_bits_equal'] and proof['saved_replies_verified']
        assert all(sha(ROOT/p)==h for p,h in proof['workflow_hashes'].items())
    d = ROOT / 'artifacts/update-f32-shapes-v1'
    d.mkdir(exist_ok=False)
    for name in ['wide64.wat','wide128.wat','delta.wat','int8_32.wat','int8_160.wat']:
        (d/name).write_bytes((upstream/name).read_bytes())
    (d/'wide512.wat').write_text((ROOT/'artifacts/f32-stack512-v1/build/wide.wat').read_text().replace('__imajev_f32_wide','__imajev_f32_wide512'))
    (d/'columns512.wat').write_bytes((ROOT/'artifacts/f32-columns512-v1/build/columns512.wat').read_bytes())
    source = (upstream/'frozen-builder.py').read_text().replace('artifacts/update-adaptive160-v2', 'artifacts/update-f32-shapes-v1')
    anchor = "    runtime = base['runtime_command'][:]"
    assert source.count(anchor) == 1
    addition = '''    p = D / 'runtime/f32_output.rs'
    text = p.read_text()
    wide_start = text.index('#[cfg(target_arch="wasm32")]\\nfn project_wide(')
    large_start = text.index('#[cfg(target_arch="wasm32")]\\nfn project_wide128(', wide_start)
    middle = text[wide_start:large_start]
    wide512 = text[large_start:].replace('128','512').replace('(marker^517)','(marker^1517)')
    columns = middle.replace('fn project_wide(', 'fn project_columns512(').replace('accumulate_wide','accumulate_columns512').replace('__imajev_f32_wide','__imajev_f32_columns512').replace('(marker^5)','(marker^1518)')
    dispatch = ' if rows%128==0 {return project_wide128(x,packed,start,n,rows,cols);}'
    assert columns.count(dispatch)==1
    columns = columns.replace(dispatch,'')
    before = 'for column in(0..cols).step_by(64)'
    assert columns.count(before)==1
    columns = columns.replace(before,'for column in(0..cols).step_by(512)')
    assert text.count(dispatch)==1
    text = text.replace(dispatch,' if rows%512==0 && cols==64 {return project_wide512(x,packed,start,n,rows,cols);}\\n if rows==64 && cols%512==0 {return project_columns512(x,packed,start,n,rows,cols);}\\n'+dispatch)
    p.write_text(text+'\\n'+wide512+'\\n'+columns)
'''
    source = source.replace(anchor, addition+anchor)
    anchor = "    sources = [Path(__file__), B / 'report.json'"
    assert source.count(anchor)==1
    extra = '''    for name,symbol in [('wide512.wat','__imajev_f32_wide512'),('columns512.wat','__imajev_f32_columns512')]:
        wat=D.parent/name
        prior=D/'full.wasm'
        result=D/'candidate.wasm'
        row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(wat),str(result),symbol],text=True))
        assert row['wasmparser_validation']
        patches.append(row)
        prior.rename(D/('before-'+name+'.wasm'))
        result.rename(D/'full.wasm')
'''
    source=source.replace(anchor,extra+anchor)
    (d/'frozen-builder.py').write_text(source)
    paths=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',d/'frozen-builder.py',
           ROOT/'artifacts/f32-stack512-v1/summary.json',ROOT/'artifacts/f32-columns512-v1/summary.json']
    paths+=list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2)+'\n')
    exec(compile(source,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':
    main()
