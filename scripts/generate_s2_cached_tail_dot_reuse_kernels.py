#!/usr/bin/env python3
"""Reuse same-A direct tail dots instead of reconstructing B inside every K dot."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 original=ROOT/'scripts/generate_s2_cached_tail_kernels.py';d=ROOT/'artifacts/s2-cached-tail-dot-reuse-kernels-v1';d.mkdir(exist_ok=True)
 s=original.read_text().replace('s2-cached-tail-kernels-v1','s2-cached-tail-dot-reuse-kernels-v1').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 begin=s.index('  for ki in range(4):\n');end=s.index('  for j in range(groups):\n   for ni in range(4):\n    offset=',begin)
 replacement='''  # Exact raw B basis identities (K-major,N-minor), verified before generation.
  programs=[[(0,[]),(4,[(0,1)]),(28,[(0,1)]),(32,[(1,1),(2,1),(0,-1)])],[(1,[]),(2,[]),(29,[(4,1)]),(30,[(5,1)])],[(7,[]),(11,[(4,1)]),(14,[]),(18,[(6,1)])],[(8,[(0,1)]),(9,[(1,1)]),(15,[(2,1)]),(16,[(3,1)])]]
  for ki in range(4):
   am,sign=BASIS['A'][ki]['terms'][0];assert len(BASIS['A'][ki]['terms'])==1 and sign==1
   direct.append(f'(local.set $qp(i32.add(i32.load offset={am*4}(local.get $q))(local.get $qoff)))')
   for j in range(groups):
    for ni,(bm,prior)in enumerate(programs[ki]):
     for ii,(slot,sgn)in enumerate(prior):
      assert ii or sgn>0
      direct.append(f'local.get $p{j}_{slot}')
      if ii:direct.append('i32x4.add'if sgn>0 else'i32x4.sub')
     for k in range(32):
      direct.append(f'(local.tee $x{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 and ni==0 else f'local.get $x{k}')
      direct +=[f'local.get $w{bm}_{j}_{k}','i32x4.dot_i16x8_s']
      if prior or k:direct.append('i32x4.add')
     slot=ni+4 if ki in [1,2]else ni
     direct.append(f'local.set $p{j}_{slot}')
    if ki in [1,2]:
     for ni in range(4):direct+=[f'local.get $p{j}_{ni}',f'local.get $p{j}_{ni+4}','i32x4.add',f'local.set $p{j}_{ni}']
'''
 s=s[:begin]+replacement+s[end:]
 s=s.replace('One-token final quartet uses16raw K/N products with cached-B basis reconstruction','One-token final quartet uses16cached-B leaf dots with integer reuse across N and exact raw basis identities')
 s=s.replace('    before=f\'(local.set $wp', '    before=f\'(local.set $wp')
 # Prove every within-K raw result using integer coefficient vectors; Ki3 carries prior K separately.
 plan=ROOT/'artifacts/s2-common-raw-kernels-v1/plan.py';ns=dict(__name__='plan',__file__=str(plan));exec(compile(plan.read_text(),str(plan),'exec'),ns);_,b,_,leaves,_,_=ns['plan']();vecs=[tuple(b.symbols[n].get(i,0)for i in range(16))for _,n in leaves]
 programs=[[(0,[]),(4,[(0,1)]),(28,[(0,1)]),(32,[(1,1),(2,1),(0,-1)])],[(1,[]),(2,[]),(29,[(0,1)]),(30,[(1,1)])],[(7,[]),(11,[(0,1)]),(14,[]),(18,[(2,1)])],[(8,[]),(9,[]),(15,[]),(16,[])]]
 for ki,program in enumerate(programs):
  done=[]
  for ni,(m,terms)in enumerate(program):
   result=tuple(vecs[m][i]+sum(sgn*done[p][i]for p,sgn in terms)for i in range(16));assert result==tuple(int(i==ki*4+ni)for i in range(16));done.append(result)
 frozen=d/'frozen-generator-entry.py';frozen.write_text(s);(d/'generator-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),original,frozen,plan]},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
