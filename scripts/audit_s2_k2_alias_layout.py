#!/usr/bin/env python3
"""Independent packed lane, compact ABI, transpose and rank49 integer reference checks."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-alias-kernels-v1';r=json.loads((d/'report.json').read_text());v=json.loads((d/'wasm-validation.json').read_text())
 assert v['all4_wasm_validated']and all(sha(ROOT/p)==h for p,h in r['source_hashes'].items())
 ns=dict(__name__='rank49',__file__=str(d/'plan.py'));exec(compile((d/'plan.py').read_text(),str(d/'plan.py'),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();rng=np.random.default_rng(712049);conditions=[]
 # Actual emitted SIMD transpose masks; compose independent integer-index shuffles.
 expected_masks=[[0,4,1,5],[0,4,1,5],[2,6,3,7],[2,6,3,7],[0,1,4,5],[2,3,6,7],[0,1,4,5],[2,3,6,7]]
 for k in r['kernels']:
  wat=(ROOT/k['path']).read_text();masks=[]
  for line in wat.splitlines():
   if line.startswith('i8x16.shuffle '):
    bytes_=list(map(int,line.split()[1:]));mask=[bytes_[i]//4 for i in range(0,16,4)]
    assert bytes_==[j*4+x for j in mask for x in range(4)];masks.append(mask)
  assert masks and all(masks[i]==expected_masks[i%8]for i in range(len(masks)))
  source=np.arange(16).reshape(4,4).T;u=[np.concatenate([source[0],source[1]])[[0,4,1,5]],np.concatenate([source[2],source[3]])[[0,4,1,5]],np.concatenate([source[0],source[1]])[[2,6,3,7]],np.concatenate([source[2],source[3]])[[2,6,3,7]]]
  out=np.concatenate([np.concatenate([u[0],u[1]])[[0,1,4,5]],np.concatenate([u[0],u[1]])[[2,3,6,7]],np.concatenate([u[2],u[3]])[[0,1,4,5]],np.concatenate([u[2],u[3]])[[2,3,6,7]]]);assert np.array_equal(out,np.arange(16))
  assert '(local.get $start)))'in wat and '(i32.const 3)'in wat
 decoded=0
 for rows,cols in [(16,256),(32,512),(80,2560),(80,9216)]:
  rr,cc=np.indices((rows,cols));m=(cc%256)//64*4+rr%4;kk=cc%64
  packed_index=m*(rows*cols//16)+(rr//16)*cols+(cc//256)*256+(kk//2)*8+((rr%16)//4)*2+kk%2
  assert np.array_equal(np.sort(packed_index.ravel()),np.arange(rows*cols));decoded+=rows*cols
  for pattern in ['random','minmax','zero','alternating']:
   raw=rng.integers(-128,128,size=(rows,cols),dtype=np.int16)if pattern=='random'else np.resize(np.array([-128,127]if pattern=='minmax'else[0]if pattern=='zero'else[-127,0,127,1,-1],dtype=np.int16),(rows,cols))
   packed=np.empty(rows*cols,dtype=np.int16);packed[packed_index]=raw
   assert np.array_equal(packed[packed_index],raw)
   for tokens in [1,3,4,5,8,9]:
    groups=(tokens+3)//4;query=np.zeros((groups*4,cols),dtype=np.int16);query[:tokens]=rng.integers(-127,128,size=(tokens,cols),dtype=np.int16)
    compact=np.empty((49,groups,cols//4),dtype=np.int16)
    for group in range(groups):
     for block in range(cols//256):
      av=query[group*4:group*4+4,block*256:(block+1)*256].reshape(16,64).astype(np.int64)
      for mi,(an,_)in enumerate(leaves):compact[mi,group,block*64:(block+1)*64]=sum(coef*av[index]for index,coef in a.symbols[an].items())
    for group in range(groups):
     # Test first and last K256 blocks, first and last 16-row output groups.
     for block in set([0,cols//256-1]):
      for output_group in set([0,rows//16-1]):
       base=output_group*cols+block*256
       # Exactly the emitted load8x8_s addresses: 32 loads, each2 K times4 lanes.
       physical=np.array([[packed[j*(rows*cols//16)+base+k*8: j*(rows*cols//16)+base+k*8+8].reshape(4,2)for k in range(32)]for j in range(16)],dtype=np.int64)
       products=[]
       for mi,(_,bn)in enumerate(leaves):
        weights=sum(coef*physical[index]for index,coef in b.symbols[bn].items());query_terms=compact[mi,group,block*64:(block+1)*64].astype(np.int64).reshape(32,2)
        products.append((weights*query_terms[:,None,:]).sum(axis=(0,2)))
       got=np.array([[sum(coef*products[index]for index,coef in c.symbols[roots[ti][ri]].items())for ri in range(4)]for ti in range(4)]).transpose(0,2,1).reshape(4,16)
       expected=query[group*4:group*4+4,block*256:(block+1)*256].astype(np.int64)@raw[output_group*16:(output_group+1)*16,block*256:(block+1)*256].astype(np.int64).T
       assert np.array_equal(got,expected),(rows,cols,pattern,tokens,group,block,output_group)
    conditions.append(dict(rows=rows,cols=cols,pattern=pattern,tokens=tokens,rank49_lane_roots_equal=True))
 files=[Path(__file__),d/'report.json',d/'wasm-validation.json',d/'plan.py']+[ROOT/k['path']for k in r['kernels']]
 report=dict(complete=True,conditions=len(conditions),decoded_weight_indices=decoded,all_integer_roots_equal=True,actual_emitted_transpose_masks_verified=True,compact_operand_shape_verified=True,raw_capacity_unchanged=True,wasm_execution_verified=False,performance_verified=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},cases=conditions,scope=__doc__)
 (d/'layout-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['source_hashes','cases']}))
if __name__=='__main__':main()
