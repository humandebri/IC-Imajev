#!/usr/bin/env python3
"""Extend validated direct output to complete aligned168 tiles."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-direct-output-v1';d=ROOT/'artifacts/update-direct168-v1';proof=ROOT/'artifacts/s1-direct168-v1/summary.json'
 r=json.loads(proof.read_text());assert r['native_bits_equal'] and r['native_digests_recomputed'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 names=[]
 for seed in [False,True]:
  name='direct168'+('-seed'if seed else'')+'.wat';text=(ROOT/'artifacts/s1-direct168-v1/build'/name).read_text();symbol='__imajev_s1_168_direct'+('_seed'if seed else'')+'_accumulate';text=text.replace('__imajev_s1_168_directseed_accumulate',symbol);assert text.count(symbol)==1;(d/name).write_text(text);names.append((name,symbol))
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-direct-output-v1','artifacts/update-direct168-v1')
 patch='''    p=D/'runtime/strassen_raw.rs';text=p.read_text()
    assert text.count('if tile!=168 && width==tile')==1
    text=text.replace('if tile!=168 && width==tile','if width==tile')
    anchor='let f=if tile==160 {if block==0{accumulate160_direct_seed}'
    assert text.count(anchor)==1
    text=text.replace(anchor,'let f=if tile==168 {if block==0{accumulate168_direct_seed}else{accumulate168_direct}}else if tile==160 {if block==0{accumulate160_direct_seed}')
    import re
    symbol='__imajev_s1_168_accumulate';fn='accumulate168'
    pat=r'\\#\\[export_name="'+symbol+r'"\\]\\#\\[inline\\(never\\)\\]\\nunsafe extern "C" fn '+fn+r'\\([^\\n]+'
    match=re.search(pat,text);assert match
    for seed in [False,True]:
        suffix='_direct_seed'if seed else'_direct';stub=match[0].replace(symbol,symbol.replace('_accumulate',suffix+'_accumulate')).replace('fn '+fn+'(','fn '+fn+suffix+'(').replace('(marker^168)','(marker^'+str(10168+(5000 if seed else 0))+')')
        text+='\\n#[cfg(target_arch="wasm32")]\\n'+stub+'\\n'
    p.write_text(text)
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 patch="    for name,symbol in "+repr(names)+":\n        wat=D.parent/name;prior=D/'full.wasm';result=D/'direct168-candidate.wasm'\n        row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(wat),str(result),symbol],text=True));assert row['wasmparser_validation'];patches.append(row);prior.rename(D/('before-'+name+'.wasm'));result.rename(D/'full.wasm')\n"
 anchor="    sources = [D/'finite-sites.json'";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['strassen_raw.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,original_arithmetic_kernels_equal=True,added_kernels=[x[1]for x in names],scope='Direct strided output for full aligned168 tiles, in addition to prior160/128/32. Positive-zero add and F32 block order preserved. Partial/unaligned views retain prior path.'),indent=2)+'\n')
if __name__=='__main__':main()
