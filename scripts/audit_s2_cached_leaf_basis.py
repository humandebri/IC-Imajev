#!/usr/bin/env python3
"""Find exact short reconstructions of original basis from49 cached Winograd leaves."""
from pathlib import Path
import hashlib,json,itertools
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'artifacts/s2-common-raw-kernels-v1/plan.py';ns=dict(__name__='plan',__file__=str(p));exec(compile(p.read_text(),str(p),'exec'),ns);a,b,_,leaves,_,_=ns['plan']();d=ROOT/'artifacts/s2-cached-leaf-basis-v1';d.mkdir(exist_ok=True);result={}
 for name,dag,names in [('A',a,[x for x,_ in leaves]),('B',b,[x for _,x in leaves])]:
  vectors=[tuple(dag.symbols[x].get(i,0)for i in range(16))for x in names];signed=[(tuple(s*v for v in vec),m,s)for m,vec in enumerate(vectors)for s in [1,-1]];lookup={vec:(m,s)for vec,m,s in reversed(signed)};cases=[]
  for i in range(16):
   target=tuple(int(j==i)for j in range(16));solution=None
   if target in lookup:solution=[lookup[target]]
   if solution is None:
    for vec,m,s in signed:
     rest=tuple(x-y for x,y in zip(target,vec))
     if rest in lookup:solution=[(m,s),lookup[rest]];break
   if solution is None:
    for (v0,m0,s0),(v1,m1,s1)in itertools.product(signed,repeat=2):
     rest=tuple(x-y-z for x,y,z in zip(target,v0,v1))
     if rest in lookup:solution=[(m0,s0),(m1,s1),lookup[rest]];break
   if solution is None:
    pairs={tuple(x+y for x,y in zip(v0,v1)):[(m0,s0),(m1,s1)]for (v0,m0,s0),(v1,m1,s1)in itertools.product(signed,repeat=2)}
    for vec,terms in pairs.items():
     rest=tuple(x-y for x,y in zip(target,vec))
     if rest in pairs:solution=terms+pairs[rest];break
   assert solution is not None,(name,i)
   assert tuple(sum(s*vectors[m][j]for m,s in solution)for j in range(16))==target
   cases.append(dict(basis=i,terms=solution,terms_count=len(solution),coefficient_identity_exact=True))
  result[name]=cases
 r=dict(complete=True,all32_basis_identities_exact=True,bases=result,source_hashes={str(x.relative_to(ROOT)):sha(x)for x in [Path(__file__),p]},scope='Symbolic exact coefficient identities for reconstructing original A/B fromcached49leaves; no generated tail Wasm, IC performance or fullpaid claim.');(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({name:[c['terms_count']for c in cases]for name,cases in result.items()}))
if __name__=='__main__':main()
