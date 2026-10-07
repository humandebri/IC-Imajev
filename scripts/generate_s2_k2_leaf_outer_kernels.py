#!/usr/bin/env python3
"""Leaf-outer rank49, 32 reused query locals, tile96 within the 10000-local cap.

All B coefficients are cached before the first quartet, then each query leaf
is shared over all output groups. Complete leaf dot order is unchanged. Leaf
products are retained per output group; C temporary registers are shared only
after all leaf dots finish. Same rank49 raw fixed payload and ABI.
"""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shuffle(mask):return 'i8x16.shuffle '+' '.join(str(i*4+b)for i in mask for b in range(4))
def kernel(ns,tile,seed):
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,capacity=ns['reconstruct'](c,roots);leafmap={n:m for m,(_,n)in enumerate(leaves)};assert len(leafmap)==49
 groups=tile//16;symbol=f'__imajev_s2_k2_{tile}'+('_seed'if seed else '')
 def cached(n,j,k):return f'w{leafmap[n]}_{j}_{k}'if n in leafmap else n
 def product(m,j):return f'p{j}_{m}'
 def register(i,j):return product(i,j)if i<49 else f'pc{i}'
 lines=[f'(module(func(export "{symbol}")'+''.join(f'(param ${p} i32)'for p in ['q','w','cols','start','sx','stride','sw','sums','n']),
 '(local $t i32)(local $qp i32)(local $yp i32)(local $qoff i32)(local $input_stride i32)(local $output_bytes i32)']
 for i in range(16):
  lines.append(f'(local $wp{i} i32)(local $bp{i} i32)')
  if f'b{i}'not in leafmap:lines.append(f'(local $b{i} v128)')
 for name,_ in b.nodes:
  if name not in leafmap:lines.append(f'(local ${name} v128)')
 for i in range(49,capacity):lines.append(f'(local $pc{i} v128)')
 for j in range(groups):
  for m in range(49):
   lines.append(f'(local ${product(m,j)} v128)')
   for k in range(32):lines.append(f'(local $w{m}_{j}_{k} v128)')
 for k in range(32):lines.append(f'(local $x{k} v128)')
 for i in range(4):lines.append(f'(local $c{i} v128)(local $u{i} v128)(local $sx{i} v128)')
 for j in range(tile//4):lines.append(f'(local $sw{j} v128)')
 for j in range(tile//4):lines.append(f'(local.set $sw{j}(v128.load offset={j*16}(local.get $sw)))')
 lines+=['(local.set $input_stride(i32.shr_u(local.get $cols)(i32.const 8)))','(local.set $output_bytes(i32.shl(local.get $stride)(i32.const 2)))']
 for i in range(16):lines.append(f'(local.set $wp{i}(i32.add(i32.load offset={i*4}(local.get $w))(local.get $start)))')
 def weights():
  out=[]
  for j in range(groups):
   for i in range(16):out.append(f'(local.set $bp{i}(i32.add(local.get $wp{i})(i32.mul(local.get $cols)(i32.const {j}))))')
   for k in range(32):
    for i in range(16):out.append(f'(local.set ${cached(f"b{i}",j,k)}(v128.load8x8_s offset={k*8}(local.get $bp{i})))')
    for name,terms in b.nodes:out+=ns['expression'](terms,lambda n:f'local.get ${cached(n,j,k)}','i16x8')+[f'local.set ${cached(name,j,k)}']
  return out
 def row():
  out=['(local.set $qoff(i32.add(i32.shr_u(i32.mul(local.get $t)(local.get $cols))(i32.const 3))(i32.shr_u(local.get $start)(i32.const 1))))']
  for ti in range(4):out.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then(local.set $sx{ti}(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $input_stride))(i32.const 2)))))))')
  for m in range(49):
   out.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qoff)))')
   for j in range(groups):
    for k in range(32):
     out.append(f'(local.tee $x{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 else f'local.get $x{k}')
     out+=[f'local.get $w{m}_{j}_{k}','i32x4.dot_i16x8_s']
     if k:out.append('i32x4.add')
    out.append(f'local.set ${product(m,j)}')
  for j in range(groups):
   out+=[re.sub(r'\$pc(\d+)',lambda m:'$'+register(int(m[1]),j),line)for line in rec]
   for ti in range(4):
    out+=[f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $output_bytes))))']
    for ri in range(4):out+=[f'local.get ${register(regs[ti][ri],j)}',f'local.set $c{ri}']
    for i,(x,y,mask)in enumerate([(0,1,[0,4,1,5]),(2,3,[0,4,1,5]),(0,1,[2,6,3,7]),(2,3,[2,6,3,7])]):out+=[f'local.get $c{x}',f'local.get $c{y}',shuffle(mask),f'local.set $u{i}']
    for part,(x,y,mask)in enumerate([(0,1,[0,1,4,5]),(0,1,[2,3,6,7]),(2,3,[0,1,4,5]),(2,3,[2,3,6,7])]):
     offset=j*64+part*16
     out+=['local.get $yp']+(['v128.const i32x4 0 0 0 0']if seed else ['local.get $yp',f'v128.load offset={offset}'])+[f'local.get $u{x}',f'local.get $u{y}',shuffle(mask),'f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{j*4+part}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
    out+=['))']
  return out
 lines+=['(block $done','(br_if $done(i32.eqz(local.get $n)))']+weights()+row()+['(local.set $t(i32.const 4))','(loop $tokens','(br_if $done(i32.ge_u(local.get $t)(local.get $n)))']+row()+['(local.set $t(i32.add(local.get $t)(i32.const 4)))','br $tokens','))','))']
 text='\n'.join(lines)+'\n';assert text.count('(')==text.count(')');count=text.count('(local $');assert count<10000,(tile,count);return text,count,symbol
def main():
 d=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1';d.mkdir(exist_ok=False);old=ROOT/'artifacts/s2-k2-alias-kernels-v1';(d/'plan.py').write_bytes((old/'plan.py').read_bytes());ns={'__name__':'plan','__file__':str(d/'plan.py')};exec(compile((d/'plan.py').read_text(),str(d/'plan.py'),'exec'),ns);kernels=[]
 for tile in [96,16]:
  for seed in [False,True]:
   text,count,symbol=kernel(ns,tile,seed);p=d/(str(tile)+('_seed'if seed else '')+'.wat');p.write_text(text);kernels.append(dict(path=str(p.relative_to(ROOT)),symbol=symbol,locals=count))
 files=[Path(__file__),old/'plan.py',d/'plan.py']+[ROOT/k['path']for k in kernels];r=dict(rank=49,kernels=kernels,query_local_vectors=32,raw_capacity_unchanged=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope=__doc__,wasm_validated=False,performance_verified=False)
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(kernels=kernels,query_local_vectors=32)))
if __name__=='__main__':main()
