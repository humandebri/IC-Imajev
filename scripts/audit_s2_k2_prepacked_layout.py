#!/usr/bin/env python3
"""Check precomputed coefficient layout, emitted loads and rank49 integer outputs."""
from pathlib import Path
import json,hashlib,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-prepacked-kernels-v1';r=json.loads((d/'report.json').read_text());ns={'__name__':'plan','__file__':str(d/'plan.py')};exec(compile((d/'plan.py').read_text(),str(d/'plan.py'),'exec'),ns);a,b,c,l,roots,_=ns['plan']();rng=np.random.default_rng(749212);cases=[]
 for rows,cols in [(16,256),(32,512),(80,2560)]:
  blocks=cols//256;w=rng.integers(-128,128,(rows,cols),dtype=np.int16);coeff=np.zeros((rows//16,blocks,49,32,4,2),dtype=np.int16)
  for g in range(rows//16):
   for block in range(blocks):
    raw=w[g*16:(g+1)*16,block*256:(block+1)*256].reshape(4,4,4,64).transpose(2,1,0,3).reshape(16,4,64)
    for m,(_,bn)in enumerate(l):coeff[g,block,m]=sum(v*raw[i].astype(np.int64)for i,v in b.symbols[bn].items()).reshape(4,32,2).transpose(1,0,2)
  flat=coeff.ravel();assert flat.nbytes==rows*cols*49//8
  for n in [1,2,3,4,5,8,9]:
   groups=(n+3)//4;q=rng.integers(-127,128,(groups*4,cols),dtype=np.int16);q[n:]=0
   for group in range(groups):
    for g in range(rows//16):
     for block in set([0,blocks-1]):
      av=q[group*4:group*4+4,block*256:(block+1)*256].reshape(16,64).astype(np.int64);products=[]
      for m,(an,_)in enumerate(l):
       x=sum(v*av[i]for i,v in a.symbols[an].items()).reshape(32,2);base=((g*blocks+block)*49+m)*256
       loaded=np.stack([flat[base+k*8:base+k*8+8].reshape(4,2)for k in range(32)]).astype(np.int64);products.append((loaded*x[:,None,:]).sum(axis=(0,2)))
      got=np.array([[sum(v*products[i]for i,v in c.symbols[roots[ti][ri]].items())for ri in range(4)]for ti in range(4)]).transpose(0,2,1).reshape(4,16)
      expected=q[group*4:group*4+4,block*256:(block+1)*256].astype(np.int64)@w[g*16:(g+1)*16,block*256:(block+1)*256].astype(np.int64).T;assert np.array_equal(got,expected)
   cases.append(dict(rows=rows,cols=cols,tokens=n,all_integer_roots_equal=True))
 for k in r['kernels']:
  text=(ROOT/k['path']).read_text();tile=int(Path(k['path']).stem.split('_')[0]);loads=re.findall(r'\(local.tee \$w(\d+)_(\d+)_(\d+)\(v128.load offset=(\d+)\(local.get \$bp0\)\)\)',text);assert len(loads)==49*32*(tile//16)
  for m,j,ki,offset in loads:assert int(offset)==int(m)*512+int(ki)*16
  assert '(i32.const 25088)'in text and '(i32.const 8)'in text
 files=[Path(__file__),d/'report.json',d/'plan.py']+[ROOT/k['path']for k in r['kernels']]
 result=dict(complete=True,all_integer_roots_equal=True,prepacked_byte_ratio=6.125,raw_capacity_unchanged=False,conditions=len(cases),actual_emitted_prepacked_offsets_verified=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},cases=cases,scope='Immutable coefficients change capacity, preserve weights/arithmetic. No performance or full-canister capacity claim.')
 (d/'layout-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['cases','source_hashes']}))
if __name__=='__main__':main()
