#!/usr/bin/env python3
"""Replace pure finite scans by the independently verified SIMD64 predicate."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-quantize-cached-v1';proof=ROOT/'artifacts/finite-simd-v1/summary.json';r=json.loads(proof.read_text());assert r['complete'] and r['all_predicates_equal'] and r['saved_candid_redecoded'] and r['selected_width']==64
 for hashes in [r['workflow_hashes'],json.loads((old/'workflow-hashes.json').read_text())]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 d=ROOT/'artifacts/update-finite-simd-v1';d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/finite_simd.rs';h=helper.read_text();assert h.count('scan::<16>(x)')==1;h=h.replace('scan::<16>(x)','scan::<64>(x)');(d/'finite_simd.rs').write_text(h)
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-finite-simd-v1');anchor="    runtime = base['runtime_command'][:]"
 patch='''    import re
    pattern=re.compile(r'([A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*(?:\\(\\))?)*)\\.iter\\(\\)\\.all\\(\\|([a-z])\\|\\s*\\2\\.is_finite\\(\\)\\)')
    finite_sites={}
    for p in (D/'runtime').glob('*.rs'):
        text=p.read_text();changed,count=pattern.subn(lambda m:'crate::all_finite(&'+m[1]+')',text)
        if count:finite_sites[p.name]=count;p.write_text(changed)
    assert sum(finite_sites.values())>=81
    (D/'finite-sites.json').write_text(json.dumps(finite_sites,indent=2)+'\\n')
    (D/'runtime/finite_simd.rs').write_bytes((D.parent/'finite_simd.rs').read_bytes())
    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\nmod finite_simd;\\npub use finite_simd::all_finite;\\n')
'''
 assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 s=s.replace("sources = [Path(__file__),", "sources = [D/'finite-sites.json',Path(__file__),")
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),old/'workflow-hashes.json',old/'frozen-builder.py',proof,helper,d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
