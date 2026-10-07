#!/usr/bin/env python3
"""Integrate verified first-block initialized sums for160/128/32; retain168 path."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-stack-store-all-v1';d=ROOT/'artifacts/update-zero-seed-v1';proofs=[ROOT/'artifacts/s1-zero-seed-v1/summary.json',ROOT/'artifacts/s1-zero-seed-mlp-v1/summary.json']
 for p in proofs:
  r=json.loads(p.read_text());assert r['native_bits_equal'] and r['native_digests_recomputed'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 for i,name in [(0,'seed128.wat'),(1,'seed160.wat'),(2,'seed32.wat')]:
  text=(ROOT/f'artifacts/s1-zero-seed-v1/build/kernel-seed-{i}.wat').read_text().replace('__imajev_seed_s1_wide_accumulate','__imajev_s1_wide_seed_accumulate').replace('__imajev_seed_s1_160_accumulate','__imajev_s1_160_seed_accumulate').replace('__imajev_seed_s1_32_accumulate','__imajev_s1_raw_seed_accumulate');(d/name).write_text(text)
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-stack-store-all-v1','artifacts/update-zero-seed-v1')
 patch=r'''    p=D/'runtime/strassen_raw.rs';text=p.read_text()
    before='let mut sums=vec![0f32;q.rows()*tile];for block in 0..cols/256'
    assert text.count(before)==1
    after='let count=q.rows()*tile;let mut initialized:Vec<f32>;let mut uninit:Vec<core::mem::MaybeUninit<f32>>;let sums_ptr=if tile==168{initialized=vec![0f32;count];initialized.as_mut_ptr()}else{uninit=Vec::with_capacity(count);unsafe{uninit.set_len(count);}uninit.as_mut_ptr().cast::<f32>()};for block in 0..cols/256'
    text=text.replace(before,after)
    a=text.index(after);b=text.index(' r+=width;}}',a);body=text[a:b].replace('sums.as_mut_ptr()','sums_ptr')
    for fn in ['accumulate160','accumulate_wide','accumulate']:
        body=body.replace(fn+'(ap.', '(if block==0{'+fn+'_seed}else{'+fn+'})(ap.')
    anchor='  for t in 0..q.rows(){out[t*rows+r..';assert body.count(anchor)==1
    body=body.replace(anchor,'  let sums=unsafe{core::slice::from_raw_parts(sums_ptr,count)};\n'+anchor)
    text=text[:a]+body+text[b:]
    import re
    for fn,symbol,marker in [('accumulate','__imajev_s1_raw_accumulate',2),('accumulate_wide','__imajev_s1_wide_accumulate',4),('accumulate160','__imajev_s1_160_accumulate',164)]:
        pat=r'\#\[export_name="'+symbol+r'"\]\#\[inline\(never\)\]\nunsafe extern "C" fn '+fn+r'\([^\n]+'
        match=re.search(pat,text);assert match,symbol
        stub=match[0].replace(symbol,symbol.replace('_accumulate','_seed_accumulate')).replace('fn '+fn+'(','fn '+fn+'_seed(').replace('(marker^'+str(marker)+')','(marker^'+str(5000+marker)+')')
        text+='\n#[cfg(target_arch="wasm32")]\n'+stub+'\n'
    p.write_text(text)
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 patch="""    for name,symbol in [('seed128.wat','__imajev_s1_wide_seed_accumulate'),('seed160.wat','__imajev_s1_160_seed_accumulate'),('seed32.wat','__imajev_s1_raw_seed_accumulate')]:
        wat=D.parent/name;prior=D/'full.wasm';result=D/'seed-candidate.wasm'
        row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(wat),str(result),symbol],text=True));assert row['wasmparser_validation'];patches.append(row);prior.rename(D/('before-'+name+'.wasm'));result.rename(D/'full.wasm')
"""
 anchor="    sources = [D/'finite-sites.json'";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),*proofs,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['strassen_raw.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,original_arithmetic_kernels_equal=True,added_kernels=['__imajev_s1_wide_seed_accumulate','__imajev_s1_160_seed_accumulate','__imajev_s1_raw_seed_accumulate'],scope='MaybeUninit sums for160/128/32, initialized in block0 with positive-zero add. Later blocks and168 path unchanged; F32 operation order preserved.'),indent=2)+'\n')
if __name__=='__main__':main()
