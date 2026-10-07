#!/usr/bin/env python3
"""Build exact update inference using the verified full unsigned-max finite scan."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-finite-simd-v1';proof=ROOT/'artifacts/finite-max-v1/summary.json';r=json.loads(proof.read_text())
 assert r['complete'] and r['all_predicates_equal'] and r['saved_candid_redecoded'] and r['queries']==318
 for hashes in [r['workflow_hashes'],json.loads((old/'workflow-hashes.json').read_text())]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 d=ROOT/'artifacts/update-finite-max-v1';d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/finite_max.rs';h=helper.read_text()+'''\npub fn all_finite(x:&[f32])->bool {
 #[cfg(target_arch="wasm32")]unsafe{return scan_max::<false>(x);}
 #[cfg(not(target_arch="wasm32"))]x.iter().all(|v|v.is_finite())
}
''';(d/'finite_simd.rs').write_text(h)
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-finite-simd-v1','artifacts/update-finite-max-v1');(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),old/'workflow-hashes.json',old/'frozen-builder.py',proof,helper,d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 before=old/'build';after=d/'build';changes=[]
 for p in (before/'runtime').glob('*.rs'):
  if p.read_bytes()!=(after/'runtime'/p.name).read_bytes():changes.append(p.name)
 assert changes==['finite_simd.rs'],changes
 assert json.loads((before/'finite-sites.json').read_text())==json.loads((after/'finite-sites.json').read_text())
 for p in old.glob('*.wat'):assert p.read_bytes()==(d/p.name).read_bytes()
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changes,predicate_sites_equal=True,arithmetic_kernels_equal=True),indent=2)+'\n')
if __name__=='__main__':main()
