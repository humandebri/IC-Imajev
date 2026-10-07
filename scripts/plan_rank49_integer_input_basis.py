#!/usr/bin/env python3
"""Reduce A preparation DAG while retaining every existing rank49 leaf/ABI."""
from pathlib import Path
from itertools import product
ROOT=Path(__file__).resolve().parents[1]
def load(p):
 ns={'__name__':'plan','__file__':str(p)};exec(compile(p.read_text(),str(p),'exec'),ns);return ns
OLD=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py'
HELPER=ROOT/'scripts/plan_rank343_integer_basis.py'
old=load(OLD);helper=load(HELPER)
def plan():
 oa,b,c,ol,roots,bound=old['plan']();a=old['DAG']('a',16)
 def coords(r,k):return (2*(r//2)+k//2,2*(r%2)+k%2)
 av={coords(r,k):f'a{r*4+k}' for r in range(4) for k in range(4)};shape=[4,4]
 for i in range(2):av,shape=helper['axis'](a,av,shape,i,helper['ta'],4)
 for i in range(2):av,shape=helper['axis'](a,av,shape,i,helper['la'],7)
 leaves=[(av[key],ol[m][1]) for m,key in enumerate(product(range(7),repeat=2))]
 assert all(a.symbols[an]==oa.symbols[oan] for (an,_),(oan,_) in zip(leaves,ol))
 assert max(sum(abs(v)for v in e.values())for e in a.symbols.values())*127<32768
 return a,b,c,leaves,roots,bound
if __name__=='__main__':
 a,b,c,l,r,_=plan();print(len(a.nodes),len(b.nodes),len(c.nodes))
