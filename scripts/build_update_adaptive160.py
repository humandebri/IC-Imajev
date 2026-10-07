#!/usr/bin/env python3
"""Combine stack F32/Delta with exact adaptive INT8 output160/128/32 tiles."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT / 'artifacts/update-f32-delta-stack-v1'
    hashes = json.loads((upstream / 'workflow-hashes.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    probe = ROOT / 'artifacts/s1-adaptive160-v1'
    check = json.loads((probe / 'summary.json').read_text())
    assert check['native_bits_equal'] and check['saved_replies_verified'] and check['ordinary_queries'] == 38
    d = ROOT / 'artifacts/update-adaptive160-v2'
    d.mkdir(exist_ok=False)
    for name in ['wide64.wat','wide128.wat','delta.wat']:
        (d / name).write_bytes((upstream / name).read_bytes())
    (d / 'int8_160.wat').write_bytes((probe / 'build/kernel.wat').read_bytes())
    (d / 'int8_32.wat').write_text((probe / 'build/kernel32.wat').read_text().replace('__imajev_s1_32_accumulate', '__imajev_s1_raw_accumulate'))
    source = (upstream / 'frozen-builder.py').read_text().replace('artifacts/update-f32-delta-stack-v1', 'artifacts/update-adaptive160-v2')
    anchor = "    runtime = base['runtime_command'][:]"
    assert source.count(anchor) == 1
    addition = '''    p = D / 'runtime/strassen_raw.rs'
    text = p.read_text()
    before = 'let tile=if cfg!(feature="experimental-strassen-output128") && rows%128==0 && start%4==0 {128}else{32};for r in(0..rows).step_by(tile){'
    assert text.count(before) == 2
    after = 'let wide=cfg!(feature="experimental-strassen-output128") && rows%128==0 && start%4==0;let mut r=0;while r<rows{let tile=if wide && rows-r>=160{160}else if wide && rows-r>=128{128}else{32};'
    text = text.replace(before,after,1)
    before = 'if tile==128 {accumulate_wide('
    assert text.count(before) == 2
    text = text.replace(before,'if tile==160 {accumulate160(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,s.as_ptr(),sums.as_mut_ptr(),q.rows());continue;}\\n   #[cfg(feature="experimental-strassen-output128")]\\n   '+before,1)
    before = 'out[t*rows+r..t*rows+r+width].copy_from_slice(&sums[t*tile..t*tile+width]);}\\n }}'
    assert text.count(before) == 2
    text = text.replace(before,before.replace('\\n }}','\\n r+=width;}}'),1)
    begin = text.index('// The128-output kernel')
    end = text.index('#[cfg(test)]',begin)
    stub = text[begin:end].replace('__imajev_s1_wide_accumulate','__imajev_s1_160_accumulate').replace('fn accumulate_wide(', 'fn accumulate160(').replace('0..n*128','0..n*160').replace('(marker^4)','(marker^164)')
    p.write_text(text[:end]+stub+text[end:])
'''
    source = source.replace(anchor, addition + anchor)
    anchor = "        assert sha(wat) == patch['source_sha256']"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + "\n        if i == 1:\n            wat = ROOT / 'artifacts/update-adaptive160-v2/int8_32.wat'")
    anchor = "    sources = [Path(__file__), B / 'report.json'"
    assert source.count(anchor) == 1
    extra = '''    wat = D.parent / 'int8_160.wat'
    prior = D / 'full.wasm'
    result = D / 'candidate.wasm'
    row = json.loads(subprocess.check_output([str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'), str(prior), str(wat), str(result), '__imajev_s1_160_accumulate'], text=True))
    assert row['wasmparser_validation']
    patches.append(row)
    prior.rename(D / 'before-int8-160.wasm')
    result.rename(D / 'full.wasm')
'''
    source = source.replace(anchor, extra + anchor)
    (d / 'frozen-builder.py').write_text(source)
    paths = [Path(__file__), upstream / 'workflow-hashes.json', upstream / 'frozen-builder.py', probe / 'summary.json',
             d / 'frozen-builder.py', d / 'int8_32.wat', d / 'int8_160.wat', d / 'wide64.wat', d / 'wide128.wat', d / 'delta.wat']
    (d / 'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + '\n')
    exec(compile(source, str(upstream / 'frozen-builder.py'), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
