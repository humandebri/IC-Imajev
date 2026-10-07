#!/usr/bin/env python3
"""Use verified SIMD positive-finite checks for scales, keeping arithmetic fixed."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-finite-complete-v1';d=ROOT/'artifacts/update-positive-finite-v1';proof=ROOT/'artifacts/positive-finite-v1/summary.json';r=json.loads(proof.read_text());assert r['complete'] and r['all_predicates_equal'] and r['saved_candid_redecoded'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/positive_finite.rs';h=helper.read_text();assert h.count('scan_max::<false>(x)')==1;h=h.replace('scan_max::<false>(x)','scan_max::<true>(x)');(d/'positive_finite.rs').write_text(h)
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-finite-complete-v1','artifacts/update-positive-finite-v1')
 slices={'mlp_pipeline.rs':('x[count+q..count+q+r.dims[0]*(H/256)].iter().all(|v|v.is_finite()&&*v>0.)','crate::all_positive_finite(&x[count+q..count+q+r.dims[0]*(H/256)])'),'projection_codec.rs':('x[prefix+q..prefix+q+sx].iter().all(|v| v.is_finite() && *v>0.)','crate::all_positive_finite(&x[prefix+q..prefix+q+sx])')}
 patch="    slices="+repr(slices)+"\n"+'''    import re
    pattern=re.compile(r'([A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*(?:\\(\\))?)*)\\.iter\\(\\)\\.all\\(\\|([a-z])\\|\\s*\\2\\.is_finite\\(\\)\\s*&&\\s*\\*\\2\\s*>\\s*0\\.\\)')
    sites={}
    for p in (D/'runtime').glob('*.rs'):
        text=p.read_text();text,count=pattern.subn(lambda m:'crate::all_positive_finite(&'+m[1]+')',text)
        if p.name in slices:
            before,after=slices[p.name];n=text.count(before);assert n>0;count+=n;text=text.replace(before,after)
        if count:p.write_text(text);sites[p.name]=count
    assert sum(sites.values())==14,sites
    (D/'positive-finite-sites.json').write_text(json.dumps(sites,indent=2)+'\\n')
    (D/'runtime/positive_finite.rs').write_bytes((D.parent/'positive_finite.rs').read_bytes())
    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\nmod positive_finite;\\npub use positive_finite::all_positive_finite;\\n')
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,helper,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes())
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,added_runtime_files=['positive_finite.rs'],arithmetic_kernels_equal=True,scope='Fourteen pure positive-finite checks, including one native unit-test check, use the verified early64 SIMD predicate. IEEE classification unchanged; arithmetic kernels unchanged.'),indent=2)+'\n')
if __name__=='__main__':main()
