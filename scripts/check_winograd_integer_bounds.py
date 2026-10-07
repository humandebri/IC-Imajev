#!/usr/bin/env python3
"""Bound every canonical bilinear intermediate after exact coefficient cancellation."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s3-winograd-v1';p=d/'frozen-generator.py';ns=dict(__file__=str(ROOT/'scripts/build_s3_winograd_probe.py'),__name__='bounds')
 exec(compile(p.read_text(),str(p),'exec'),ns);a,b,c,leaves,roots,old=ns['plan']()
 polynomials={};leaf_bounds=[]
 for m,(an,bn) in enumerate(leaves):
  av=np.zeros(64,dtype=np.int64);bv=np.zeros(64,dtype=np.int64)
  for i,v in a.symbols[an].items():av[i]=v
  for i,v in b.symbols[bn].items():bv[i]=v
  poly=av[:,None]*bv[None,:];polynomials[f'p{m}']=poly
  leaf_bounds.append(int(np.abs(poly).sum())*127*128*32)
 assert max(leaf_bounds)<2**31
 bounds=[]
 for name,terms in c.nodes:
  poly=np.zeros((64,64),dtype=np.int64)
  for parent,sign in terms.items():poly+=sign*polynomials[parent]
  polynomials[name]=poly;bound=int(np.abs(poly).sum())*127*128*32
  assert bound<2**31,(name,bound)
  bounds.append(dict(name=name,bound=bound,monomial_l1=int(np.abs(poly).sum())))
 for i,row in enumerate(roots):
  for j,name in enumerate(row):
   expected=np.zeros((64,64),dtype=np.int64)
   for k in range(8):expected[i*8+k,k*8+j]=1
   assert np.array_equal(polynomials[name],expected)
 result=dict(rank=343,all_651_reconstruction_nodes_checked=len(bounds)==651,max_leaf_bound=max(leaf_bounds),max_canonical_reconstruction_bound=max(v['bound'] for v in bounds),original_triangle_bound=old,final_full_dot_bound=256*127*128,all_i32_intermediates_fit=True,all_final_roots_are_original_dots=True,bounds=bounds,source_hashes={str(p.relative_to(ROOT)):sha(p),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},scope='Universal coefficient bounds for q in[-127,127], raw weights in[-128,127] and32 columns per leaf; every running leaf sum is bounded by the full absolute sum. Exact I16 transforms are separately checked by plan(). Canonical bilinear cancellation proves I32 does not wrap, despite the loose sum over transformed leaves exceeding signed32.')
 (d/'integer-bounds.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['bounds','source_hashes','scope']})
if __name__=='__main__':main()
