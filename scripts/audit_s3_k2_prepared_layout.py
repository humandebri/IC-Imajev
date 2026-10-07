#!/usr/bin/env python3
"""Independent ascending-K oracle and emitted register/shuffle/address audit."""
from pathlib import Path
import hashlib,json,re,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/s3-k2-prepared-kernels-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'report.json').read_text());assert all(sha(ROOT/p)==v for p,v in r['source_hashes'].items())
 ns={'__name__':'plan','__file__':str(D/'plan.py')};exec(compile((D/'plan.py').read_text(),str(D/'plan.py'),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,_=ns['reconstruct'](c,roots)
 rng=np.random.default_rng(343232);cases=[];overflow_nodes=0
 # Check the exact emitted C register program, including wraparound semantics.
 def reconstruct(products):
  nonlocal overflow_nodes
  pc={i:p.astype(np.uint32)for i,p in enumerate(products)};stack=[]
  for line in rec:
   if line.startswith('local.get $pc'):stack.append(pc[int(line.removeprefix('local.get $pc'))])
   elif line.startswith('local.set $pc'):pc[int(line.removeprefix('local.set $pc'))]=stack.pop()
   elif line in ['i32x4.add','i32x4.sub']:
    y=stack.pop();x=stack.pop();wide=x.astype(np.int64)+(y.astype(np.int64)if line.endswith('add')else -y.astype(np.int64));overflow_nodes+=int(np.any((wide<0)|(wide>2**32-1)));stack.append(wide.astype(np.uint32))
   elif line=='(v128.const i32x4 0 0 0 0)':stack.append(np.zeros(4,dtype=np.uint32))
   else:raise AssertionError(line)
  assert not stack
  return pc
 def emitted_outputs(pc,ti,text):
  # Parse the actual emitted three byte-shuffles for every four-output store.
  fragment=text.split(f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $output_bytes))))')[1]
  got=np.zeros(32,dtype=np.int32)
  parts=fragment.split('f32x4.convert_i32x4_s')[:8]
  for index,part in enumerate(parts):
   references=[int(x)for x in re.findall(r'local.get \$pc(\d+)',part)][-4:]
   masks=re.findall(r'i8x16.shuffle ((?:\d+ ){15}\d+)',part)[-3:];assert len(references)==4 and len(masks)==3
   def sh(x,y,mask):return np.concatenate([x.view(np.uint8),y.view(np.uint8)])[list(map(int,mask.split()))].copy().view(np.int32)
   u=sh(pc[references[0]],pc[references[1]],masks[0]);v=sh(pc[references[2]],pc[references[3]],masks[1]);z=sh(u,v,masks[2]);got[index*4:index*4+4]=z
  return got
 texts=[(ROOT/k['path']).read_text()for k in r['kernels']]
 for text in texts:
  loads=re.findall(r'\(local.tee \$w(\d+)_(\d+)\(v128.load offset=(\d+)\(local.get \$bp\)\)\)',text);assert len(loads)==5488
  for m,k,offset in loads:assert int(offset)==int(m)*256+int(k)*16
  assert '\n'.join(rec)in text;assert '(i32.const 87808)'in text
  assert [int(o)for o in re.findall(r'v128.store offset=(\d+)',text)]==list(range(0,128,16))*16
 for cols in [256,512,2560]:
  blocks=cols//256;w=rng.integers(-128,128,(32,cols),dtype=np.int16)
  if cols==256:w[:]=np.where(np.indices(w.shape)[1]%2,-128,127)
  sw=rng.uniform(.001,.1,32).astype(np.float32)
  coeff=np.empty((blocks,343,16,4,2),dtype=np.int16)
  for block in range(blocks):
   raw=w[:,block*256:(block+1)*256].reshape(4,8,8,32).transpose(2,1,0,3).reshape(64,4,32).astype(np.int64)
   for m,(_,bn)in enumerate(leaves):
    value=sum(v*raw[i]for i,v in b.symbols[bn].items());assert np.max(np.abs(value))<=8192;coeff[block,m]=value.reshape(4,16,2).transpose(1,0,2)
  flat=coeff.ravel();assert flat.nbytes==32*cols*343//32
  for n in [0,1,2,3,7,8,9,15,16,17]:
   groups=(n+7)//8;q=rng.integers(-127,128,(groups*8,cols),dtype=np.int16);q[n:]=0
   if n==8 and cols==256:q[:]=np.where(np.indices(q.shape)[1]%2,-127,127)
   sx=rng.uniform(.001,.1,(groups*8,blocks)).astype(np.float32)
   for seed in [False,True]:
    initial=rng.standard_normal((n,40)).astype(np.float32);initial[:,::2]=-0.0
    actual=initial.copy();expected=initial.copy()
    for block in range(blocks):
     for group in range(groups):
      av=q[group*8:group*8+8,block*256:(block+1)*256].reshape(64,32).astype(np.int64);products=[]
      for m,(an,_)in enumerate(leaves):
       x=sum(v*av[i]for i,v in a.symbols[an].items());assert np.max(np.abs(x))<=8128;x=x.reshape(16,2)
       # Flat offsets match the actual first-dot v128 loads, not a matrix shortcut.
       start=(block*343+m)*128
       loaded=np.stack([flat[start+k*8:start+k*8+8].reshape(4,2)for k in range(16)]).astype(np.int64)
       products.append((loaded*x[:,None,:]).sum(axis=(0,2)))
      pc=reconstruct(products)
      for ti in range(min(8,n-group*8)):
       row=group*8+ti;got=emitted_outputs(pc,ti,texts[int(seed)])
       oracle=q[row,block*256:(block+1)*256].astype(np.int64)@w[:,block*256:(block+1)*256].astype(np.int64).T
       assert np.array_equal(got,oracle),(cols,n,seed,block,ti)
       term=np.multiply(np.multiply(got.astype(np.float32),sx[row,block],dtype=np.float32),sw,dtype=np.float32)
       reference_term=np.multiply(np.multiply(oracle.astype(np.float32),sx[row,block],dtype=np.float32),sw,dtype=np.float32)
       carry=np.zeros(32,dtype=np.float32)if seed and block==0 else actual[row,:32]
       reference_carry=np.zeros(32,dtype=np.float32)if seed and block==0 else expected[row,:32]
       actual[row,:32]=np.add(carry,term,dtype=np.float32);expected[row,:32]=np.add(reference_carry,reference_term,dtype=np.float32)
    assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
    cases.append(dict(cols=cols,tokens=n,seed=seed,integer_roots_equal=True,f32_bits_equal=True,padding_preserved=True))
 assert overflow_nodes>0
 files=[Path(__file__),D/'report.json',D/'plan.py']+[ROOT/k['path']for k in r['kernels']]
 result=dict(complete=True,cases=cases,conditions=len(cases),reconstruction_wraparound_operations=overflow_nodes,emitted_load_and_shuffle_offsets_checked=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Host layout/register/bit oracle only. No executed Wasm, inference timing or full goal claim.')
 (D/'layout-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(D/'layout-evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[D/'layout-audit.json']):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps({k:v for k,v in result.items()if k not in ['cases','source_hashes']}))
if __name__=='__main__':main()
