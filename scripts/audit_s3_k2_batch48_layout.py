#!/usr/bin/env python3
"""Independent scalar dots, modulo reconstruction, and all batch/tile offsets."""
from pathlib import Path
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/s3-k2-batch48-kernels-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'report.json').read_text())
 for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p
 ns={'__name__':'plan','__file__':str(D/'plan.py')};exec(compile((D/'plan.py').read_text(),str(D/'plan.py'),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,_=ns['reconstruct'](c,roots)
 for item in r['kernels']:
  text=(ROOT/item['path']).read_text()
  loads=re.findall(r'\(local.set \$w(\d+)_(\d+)\(v128.load offset=(\d+)\(local.get \$bp(\d+)\)\)\)',text)
  assert len(loads)==343*4*16
  for index,(j,k,offset,bp)in enumerate(loads):assert int(j)==int(bp) and int(offset)==(index//64)*256+int(k)*16
  for ti in range(6):
   for j in range(4):
    expected='\n'.join(re.sub(r'\$pc(\d+)',lambda m:f'$p{ti}_{j}_{m[1]}'if int(m[1])<343 else m[0],line)for line in rec)
    assert text.count(expected)==1
    for m in range(343):assert text.count(f'local.set $p{ti}_{j}_{m}\n')==1
  offsets=list(map(int,re.findall(r'v128.store offset=(\d+)',text)))
  assert offsets==[j*128+o for ti in range(6) for j in range(4) for row in range(8) for o in range(0,128,16)]
  assert text.count('(local.set $t(i32.add(local.get $t)(i32.const 48)))')==1
 rng=np.random.default_rng(34348128);cases=[]
 for cols in [256,512]:
  w=rng.integers(-128,128,(128,cols),dtype=np.int16);sw=rng.uniform(.001,.1,128).astype(np.float32)
  for n in [0,1,7,8,9,47,48,49,56,57,96,97]:
   q=rng.integers(-127,128,(((n+7)//8)*8,cols),dtype=np.int16);q[n:]=0
   sx=rng.uniform(.001,.1,(len(q),cols//256)).astype(np.float32)
   for seed in [False,True]:
    initial=rng.standard_normal((n,132)).astype(np.float32);initial[:,::2]=-0.0
    actual=initial.copy();expected=initial.copy()
    for block in range(cols//256):
     raw=w[:,block*256:(block+1)*256].reshape(4,4,8,8,32).transpose(3,2,0,1,4).reshape(64,4,4,32).astype(np.int64)
     bs=[sum(v*raw[i]for i,v in b.symbols[bn].items())for _,bn in leaves]
     for t in range(0,n,48):
      count=min(48,n-t);groups=(count+7)//8
      av=q[t:t+groups*8,block*256:(block+1)*256].reshape(groups,8,8,32).transpose(1,2,0,3).reshape(64,groups,32).astype(np.int64)
      pc={}
      for m,(an,_)in enumerate(leaves):
       x=sum(v*av[i]for i,v in a.symbols[an].items());assert np.max(np.abs(x))<=4064 and np.max(np.abs(bs[m]))<=4096
       # Each sequential two-K dot feeds a wrapping I32 accumulation.
       acc=np.zeros((groups,4,4),dtype=np.uint32)
       for k in range(16):
        term=(x[:,None,None,k*2:k*2+2]*bs[m][None,:,:,k*2:k*2+2]).sum(axis=-1)
        acc=(acc.astype(np.int64)+term).astype(np.uint32)
       pc[m]=acc
      stack=[]
      for line in rec:
       if line.startswith('local.get'):stack.append(pc[int(line.split('$pc')[1])])
       elif line.startswith('local.set'):pc[int(line.split('$pc')[1])]=stack.pop()
       elif line.startswith('(v128.const'):stack.append(np.zeros((groups,4,4),dtype=np.uint32))
       else:
        right=stack.pop();left=stack.pop();stack.append((left.astype(np.int64)+(right.astype(np.int64)if line.endswith('add')else -right.astype(np.int64))).astype(np.uint32))
      assert not stack
      for row in range(count):
       group,ti=divmod(row,8)
       # Root vectors carry the four independent output lanes in original layout.
       got=np.stack([pc[regs[ti][ni]][group].view(np.int32)for ni in range(8)],axis=-1).reshape(128)
       oracle=q[t+row,block*256:(block+1)*256].astype(np.int64)@w[:,block*256:(block+1)*256].astype(np.int64).T
       assert np.array_equal(got,oracle)
       term=np.multiply(np.multiply(got.astype(np.float32),sx[t+row,block],dtype=np.float32),sw,dtype=np.float32)
       reference=np.multiply(np.multiply(oracle.astype(np.float32),sx[t+row,block],dtype=np.float32),sw,dtype=np.float32)
       carry=np.zeros(128,dtype=np.float32)if seed and block==0 else actual[t+row,:128]
       prior=np.zeros(128,dtype=np.float32)if seed and block==0 else expected[t+row,:128]
       actual[t+row,:128]=np.add(carry,term,dtype=np.float32);expected[t+row,:128]=np.add(prior,reference,dtype=np.float32)
    assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
    cases.append(dict(cols=cols,tokens=n,seed=seed,all_integer_and_f32_bits_equal=True))
 files=[Path(__file__),D/'report.json',D/'plan.py']+[ROOT/k['path']for k in r['kernels']]
 report=dict(complete=True,conditions=len(cases),cases=cases,all_24_product_and_reconstruction_mappings_checked=True,
   all_weight_and_store_offsets_checked=True,wasm_execution_verified=False,performance_verified=False,
   limitation='Host exact dots/register recurrence and emitted mapping/offset audit; actual Wasm shuffle execution and performance remain unverified.',
   source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (D/'layout-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=True,conditions=len(cases))))
if __name__=='__main__':main()
