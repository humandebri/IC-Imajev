#!/usr/bin/env python3
"""Independent raw-layout oracle plus emitted leaf and C-temporary lifetimes."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1';report=json.loads((d/'report.json').read_text());ns={'__name__':'plan','__file__':str(d/'plan.py')};exec(compile((d/'plan.py').read_text(),str(d/'plan.py'),'exec'),ns);_,_,c,_,roots,_=ns['plan']();rec,_,capacity=ns['reconstruct'](c,roots)
 for item in report['kernels']:
  text=(ROOT/item['path']).read_text();tile=int(Path(item['path']).stem.split('_')[0]);groups=tile//16
  assert len(re.findall(r'\(local \$x\d+ v128\)',text))==32
  for m in range(49):
   assert text.count(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qoff)))')==2
   for j in range(groups):
    needle=f'local.set $p{j}_{m}';assert text.count(needle+'\n')==2
  for j in range(groups):
   expected='\n'.join(re.sub(r'\$pc(\d+)',lambda m:'$'+(f'p{j}_{int(m[1])}'if int(m[1])<49 else f'pc{m[1]}'),line)for line in rec)
   assert text.count(expected)==2
  assert all(i>=49 for i in map(int,re.findall(r'^local.set \$pc(\d+)$',text,re.M)))
  assert text.index('(local.set $bp0')<text.index('(local.set $qoff')
 old=ROOT/'scripts/audit_s2_k2_layout.py';source=old.read_text().replace('s2-k2-kernels-v1','s2-k2-leaf-outer-kernels-v1').replace('(80,2560),(80,9216)','(96,2560),(96,9216)')
 p=d/'frozen-layout-auditor.py';assert not p.exists();p.write_text(source)
 (d/'auditor-entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n')
 exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 r=json.loads((d/'layout-audit.json').read_text());r.update(shared_C_temporaries_after_all_leaf_dots_verified=True,emitted_all_group_leaf_assignments_verified=True,query_local_vectors=32,new_output_tile=96,scope='Host independent integer/layout/address/mask and emitted register program checks. Actual Wasm outputs and performance still unverified.');(d/'layout-audit.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
