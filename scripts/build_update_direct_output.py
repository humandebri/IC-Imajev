#!/usr/bin/env python3
"""Integrate exact direct output for full aligned160/128/32 tiles; preserve fallbacks."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-zero-seed-v1';d=ROOT/'artifacts/update-direct-output-v1';proofs=[ROOT/'artifacts/s1-direct-output-v1/summary.json',ROOT/'artifacts/s1-direct-output-mlp-v1/summary.json']
 for p in proofs:
  r=json.loads(p.read_text());assert r['native_bits_equal'] and r['native_digests_recomputed'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 names=[]
 for i,tile in enumerate([128,160,32]):
  kind='wide'if tile==128 else'160'if tile==160 else'raw'
  for seed,prefix in [(False,'__imajev_direct_'),(True,'__imajev_directseed_')]:
   text=(ROOT/f'artifacts/s1-direct-output-v1/build/direct-{i}-{prefix}.wat').read_text();oldsymbol=f'{prefix}s1_'+('wide'if tile==128 else str(tile))+'_accumulate';symbol=f'__imajev_s1_{kind}_direct'+('_seed'if seed else'')+'_accumulate';assert text.count(oldsymbol)==1
   name=f'direct{tile}'+('-seed'if seed else'')+'.wat';(d/name).write_text(text.replace(oldsymbol,symbol));names.append((name,symbol))
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-zero-seed-v1','artifacts/update-direct-output-v1')
 patch='    p=D/\'runtime/strassen_raw.rs\';text=p.read_text()\n    anchor=\'  let count=q.rows()*tile;let mut initialized:Vec<f32>;\'\n    assert text.count(anchor)==1\n    direct=\'\'\'  if tile!=168 && width==tile && rows%32==0 && (start+r)%4==0 {\n   for block in 0..cols/256 {unsafe{\n    let f=if tile==160 {if block==0{accumulate160_direct_seed}else{accumulate160_direct}}else if tile==128 {if block==0{accumulate_wide_direct_seed}else{accumulate_wide_direct}}else {if block==0{accumulate_direct_seed}else{accumulate_direct}};\n    f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),rows,s.as_ptr(),out.as_mut_ptr().add(r),q.rows());\n   }}\n   r+=width;continue;\n  }\n\'\'\'\n    text=text.replace(anchor,direct+anchor)\n    import re\n    for fn,symbol,marker in [(\'accumulate\',\'__imajev_s1_raw_accumulate\',2),(\'accumulate_wide\',\'__imajev_s1_wide_accumulate\',4),(\'accumulate160\',\'__imajev_s1_160_accumulate\',164)]:\n        pat=r\'\\#\\[export_name="\'+symbol+r\'"\\]\\#\\[inline\\(never\\)\\]\\nunsafe extern "C" fn \'+fn+r\'\\([^\\n]+\'\n        match=re.search(pat,text);assert match,symbol\n        for seed in [False,True]:\n            suffix=\'_direct_seed\'if seed else\'_direct\';stub=match[0].replace(symbol,symbol.replace(\'_accumulate\',suffix+\'_accumulate\')).replace(\'fn \'+fn+\'(\',\'fn \'+fn+suffix+\'(\').replace(\'(marker^\'+str(marker)+\')\',\'(marker^\'+str(10000+marker+(5000 if seed else 0))+\')\')\n            text+=\'\\n#[cfg(target_arch="wasm32")]\\n\'+stub+\'\\n\'\n    p.write_text(text)\n'
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 patch="    for name,symbol in "+repr(names)+":\n        wat=D.parent/name;prior=D/'full.wasm';result=D/'direct-candidate.wasm'\n        row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(wat),str(result),symbol],text=True));assert row['wasmparser_validation'];patches.append(row);prior.rename(D/('before-'+name+'.wasm'));result.rename(D/'full.wasm')\n"
 anchor="    sources = [D/'finite-sites.json'";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),*proofs,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['strassen_raw.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,original_arithmetic_kernels_equal=True,added_kernels=[x[1]for x in names],scope='Direct strided output for full aligned160/128/32 tiles. Positive-zero add and F32 block order preserved. Partial/unaligned views and168 retain prior path.'),indent=2)+'\n')
if __name__=='__main__':main()
