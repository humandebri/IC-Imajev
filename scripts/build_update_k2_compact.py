#!/usr/bin/env python3
"""Full runtime K2 exact rank7, compact operands, single and carry paths."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/update-gated-norm-finalize-v1';probe=ROOT/'artifacts/s1-k2-mlp160-v1';d=ROOT/'artifacts/update-k2-compact-v1';d.mkdir(exist_ok=False)
 audit=json.loads((probe/'post-report-audit.json').read_text());assert audit['old_28_kernel_sources_unchanged']and len(audit['completed'])==2
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 (d/'strassen_k2_runtime.rs').write_bytes((ROOT/'scripts/strassen_k2_runtime.rs').read_bytes())
 text=(probe/'build/src/win_kernel.rs').read_text()
 counter=[32768]
 def distinct(match):
  counter[0]+=1
  return 'f32::from_bits((marker^'+str(counter[0])+')|0x7fc00000)'
 import re
 text=re.sub(r'f32::from_bits\(marker\|0x7fc00000\)',distinct,text)
 assert counter[0]==32768+9
 (d/'win_kernel.rs').write_text(text)
 for tile in [128,168,160,32]:
  for seed in [False,True]:
   name=f'direct{tile}'+('_seed'if seed else '')+'.wat';(d/('k2-'+name)).write_bytes((probe/'build'/name).read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-gated-norm-finalize-v1','artifacts/update-k2-compact-v1')
 patch='''    p=D/'runtime/strassen_raw.rs';text=p.read_text();stubs=[]
    for line in text.splitlines():
        if line.startswith('unsafe extern "C" fn '):
            symbol=text[:text.index(line)].rsplit('#[export_name="',1)[1].split('"',1)[0]
            stubs.append('#[cfg(target_arch="wasm32")]\\n#[export_name="'+symbol+'"]#[inline(never)]\\n'+line)
    assert len(stubs)==15
    p.write_text((D.parent/'strassen_k2_runtime.rs').read_text()+'\\n'+'\\n'.join(stubs)+'\\n')
    (D/'runtime/win_kernel.rs').write_bytes((D.parent/'win_kernel.rs').read_bytes())
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 patch='''    for tile in [128,168,160,32]:
        for seed in [False,True]:
            symbol=('__imajev_win7_wide_accumulate'if tile==128 else f'__imajev_win7_{tile}_accumulate')+('_seed'if seed else '')
            wat=D.parent/(f'k2-direct{tile}'+('_seed'if seed else '')+'.wat');prior=D/'full.wasm';result=D/'k2-patched.wasm'
            row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(wat),str(result),symbol],text=True));assert row['wasmparser_validation'];patches.append(row);prior.rename(D/(f'before-k2-{tile}'+('-seed'if seed else '')+'.wasm'));result.rename(D/'full.wasm')
'''
 anchor='    sources = ';at=s.index(anchor);s=s[:at]+patch+s[at:]
 s=s.replace("report = dict(baseline=","sources += list(D.parent.glob('*.wat'))+[D.parent/'strassen_k2_runtime.rs',D.parent/'win_kernel.rs']\n    report = dict(baseline=")
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),ROOT/'scripts/strassen_k2_runtime.rs',old/'frozen-builder.py',probe/'post-report-audit.json',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
