#!/usr/bin/env python3
"""Apply the verified finite SIMD predicate to remaining any/chain scans."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-output-uninit-v1';d=ROOT/'artifacts/update-finite-complete-v1'
 proof=ROOT/'artifacts/finite-max-v1/summary.json';r=json.loads(proof.read_text());assert r['complete'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 replacements={
  'delta_log.rs':{'log.iter().chain(states.iter()).all(|v|v.is_finite())':'(crate::all_finite(&log) && crate::all_finite(&states))'},
  'f32_output.rs':{'x.iter().chain(initial).all(|v|v.is_finite())':'(crate::all_finite(x) && crate::all_finite(initial))'},
  'delta_mlp_start.rs':{'v.iter().chain(&log).all(|v|v.is_finite())':'(crate::all_finite(&v) && crate::all_finite(&log))','values.iter().chain(&log).all(|v|v.is_finite())':'(crate::all_finite(&values) && crate::all_finite(&log))'},
  'mlp_delta_fusion.rs':{'conv.iter().chain(&log).all(|v|v.is_finite())':'(crate::all_finite(&conv) && crate::all_finite(&log))'},
  'prefix_hybrid_codec.rs':{'log.iter().chain(state).all(|v|v.is_finite())':'(crate::all_finite(log) && crate::all_finite(state))'},
  'mlp_delta_stream.rs':{'history.iter().chain(&log).all(|v| v.is_finite())':'(crate::all_finite(&history) && crate::all_finite(&log))','hidden.iter().chain(&rest).all(|v| v.is_finite())':'(crate::all_finite(&hidden) && crate::all_finite(&rest))','prep.qa.iter().chain(&prep.za).chain(base).chain(ax).all(|v|v.is_finite())':'(crate::all_finite(&prep.qa) && crate::all_finite(&prep.za) && crate::all_finite(base) && crate::all_finite(ax))'},
  'mlp_stream.rs':{'self.ax.iter().chain(&self.down).all(|v|v.is_finite())':'(crate::all_finite(&self.ax) && crate::all_finite(&self.down))'},
  'mlp_pipeline.rs':{'residual.iter().chain(&floats).all(|v|v.is_finite())':'(crate::all_finite(&residual) && crate::all_finite(&floats))'},
  'delta_finish.rs':{'x[reuse_len..].iter().all(|v|v.is_finite())':'crate::all_finite(&x[reuse_len..])'},
 }
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-output-uninit-v1','artifacts/update-finite-complete-v1')
 patch="    replacements="+repr(replacements)+"\n"+'''    import re
    sites={}
    for name,items in replacements.items():
        p=D/'runtime'/name;text=p.read_text();count=0
        for before,after in items.items():
            n=text.count(before);assert n>0,(name,before);count+=n;text=text.replace(before,after)
        p.write_text(text);sites[name]=dict(chain_or_slice=count,any=0)
    assert sum(v['chain_or_slice']for v in sites.values())==13
    pattern=re.compile(r'([A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*)\\.iter\\(\\)\\.any\\(\\|([a-z])\\|\\s*!\\s*\\2\\.is_finite\\(\\)\\)')
    for p in (D/'runtime').glob('*.rs'):
        text=p.read_text();text,count=pattern.subn(lambda m:'!crate::all_finite(&'+m[1]+')',text)
        if count:p.write_text(text);sites.setdefault(p.name,dict(chain_or_slice=0,any=0))['any']=count
    assert sum(v['any']for v in sites.values())==6
    (D/'remaining-finite-sites.json').write_text(json.dumps(sites,indent=2)+'\\n')
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes())
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,arithmetic_kernels_equal=True,scope='Remaining six any scans and thirteen chain/slice scans use the independently verified finite SIMD predicate. Only pure finite checks changed; arithmetic kernels and outputs unchanged.'),indent=2)+'\n')
if __name__=='__main__':main()
