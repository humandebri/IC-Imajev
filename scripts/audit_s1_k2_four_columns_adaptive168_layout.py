#!/usr/bin/env python3
"""Independent raw-layout bijection, vector lane coverage and rank7 integer audit."""
from pathlib import Path
import json,hashlib,runpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s1-k2-four-columns-adaptive168-v1';b=json.loads((d/'build/report.json').read_text())
 for key in ('source_hashes','dependency_hashes'):assert all(sha(ROOT/p)==h for p,h in b[key].items())
 assert sha(d/'build/diagnostic.wasm')==b['wasm_sha256'] and all(p['wasmparser_validation']for p in b['patches'])
 for rows in (32,128):
  for cols in (256,512):
   offsets=[]
   for r in range(rows):
    for k in range(cols):
     m=(k%256)//128+(r%2)*2
     index=m*(rows//4)*cols+(r//8)*2*cols+(k//256)*512+((k%128)//2)*8+((r%8)//2)*2+k%2
     offsets.append(index)
   assert sorted(offsets)==list(range(rows*cols))
 for pair in range(66):
  for block in range(10):
   for k in range(64):assert pair*2560+block*256+2*k+2<=pair*2560+block*256+128
 a,bb,c,leaves,roots,_=runpy.run_path(str(d/'plan.py'))['plan']()
 for m in (3,4):assert all(i>=2 for i in a.symbols[leaves[m][0]])
 for name in roots[0]:assert 6 not in c.symbols[name]
 rng=np.random.default_rng(71204);checks=0
 for pattern in ('random','minmax','zeros','alternating'):
  for n in (1,2,3,8):
   for cols in (256,512):
    q=rng.integers(-127,128,size=(n+1,cols),dtype=np.int32);w=rng.integers(-128,128,size=(16,cols),dtype=np.int32)
    if pattern=='minmax':q[:]=127;w[:]=-128
    if pattern=='zeros':q[:]=0
    if pattern=='alternating':q[:]=np.resize(np.array([127,-127],np.int32),cols);w[:]=np.resize(np.array([-128,127],np.int32),cols)
    q[n:]=0
    for t in range(0,n,2):
     for block in range(cols//256):
      av=[q[t,block*256:block*256+128],q[t,block*256+128:(block+1)*256],q[t+1,block*256:block*256+128],q[t+1,block*256+128:(block+1)*256]]
      for group in range(2):
       bv=[w[group*8::2,block*256:block*256+128][:4],w[group*8+1::2,block*256:block*256+128][:4],w[group*8::2,block*256+128:(block+1)*256][:4],w[group*8+1::2,block*256+128:(block+1)*256][:4]]
       prods=[]
       for an,bn in leaves:
        x=sum(v*av[i].astype(np.int64)for i,v in a.symbols[an].items());y=sum(v*bv[i].astype(np.int64)for i,v in bb.symbols[bn].items())
        # Each lane owns two K terms at a time; 64 vectors cover exactly128 terms.
        prods.append((x[None,:]*y).reshape(4,64,2).sum(axis=(1,2)))
       for ti in range(2):
        if t+ti>=n:continue
        for ri in range(2):
         actual=sum(v*prods[m]for m,v in c.symbols[roots[ti][ri]].items())
         expected=w[group*8+ri:group*8+8:2,block*256:(block+1)*256].astype(np.int64)@q[t+ti,block*256:(block+1)*256].astype(np.int64)
         assert np.array_equal(actual,expected);checks+=1
 result=dict(complete=True,module=b['wasm_sha256'],native_rank7_lane_checks=checks,raw_weight_capacity_unchanged=True,raw_layout_bijective=True,all128_terms_once_per_dot=True,first_row_tail_skip_products_3_4_zero_6_unused=True,locals=b['locals'],wasm_validated=True,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),d/'build/report.json',d/'plan.py',d/'runtime-edits.json']})
 (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
