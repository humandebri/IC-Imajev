#!/usr/bin/env python3
"""Generate rank49 K2 direct/seed kernels, four independent output quartets per lane group."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def shuffle(mask):return 'i8x16.shuffle '+' '.join(str(i*4+b)for i in mask for b in range(4))
def kernel(ns,tile,seed):
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,capacity=ns['reconstruct'](c,roots)
 symbol=f'__imajev_s2_k2_{tile}'+('_seed'if seed else '')
 lines=[f'(module(func(export "{symbol}")(param $q i32)(param $w i32)(param $cols i32)(param $start i32)(param $sx i32)(param $stride i32)(param $sw i32)(param $sums i32)(param $n i32)', '(local $t i32)(local $qp i32)(local $yp i32)(local $qoff i32)(local $input_stride i32)(local $output_bytes i32)']
 for i in range(16):lines.append(f'(local $wp{i} i32)(local $bp{i} i32)(local $b{i} v128)')
 for name,_ in b.nodes:lines.append(f'(local ${name} v128)')
 for i in range(capacity):lines.append(f'(local $pc{i} v128)')
 for m in range(49):
  for k in range(32):lines.append(f'(local $x{m}_{k} v128)')
  for j in range(tile//16):
   for k in range(32):lines.append(f'(local $w{m}_{j}_{k} v128)')
 for ri in range(4):lines.append(f'(local $c{ri} v128)(local $u{ri} v128)(local $sx{ri} v128)')
 for j in range(tile//4):lines.append(f'(local $sw{j} v128)')
 for j in range(tile//4):lines.append(f'(local.set $sw{j}(v128.load offset={j*16}(local.get $sw)))')
 lines+=['(local.set $input_stride(i32.shr_u(local.get $cols)(i32.const 8)))','(local.set $output_bytes(i32.shl(local.get $stride)(i32.const 2)))']
 for i in range(16):lines.append(f'(local.set $wp{i}(i32.add(i32.load offset={i*4}(local.get $w))(local.get $start)))')
 def row(first):
  out=['(local.set $qoff(i32.add(i32.shr_u(i32.mul(local.get $t)(local.get $cols))(i32.const 3))(i32.shr_u(local.get $start)(i32.const 1))))']
  for ti in range(4):out.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then(local.set $sx{ti}(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $input_stride))(i32.const 2)))))))')
  for j in range(tile//16):
   if first:
    for bi in range(16):out.append(f'(local.set $bp{bi}(i32.add(local.get $wp{bi})(i32.mul(local.get $cols)(i32.const {j}))))')
    for k in range(32):
     for bi in range(16):out.append(f'(local.set $b{bi}(v128.load8x8_s offset={k*8}(local.get $bp{bi})))')
     for name,terms in b.nodes:out+=ns['expression'](terms,lambda n:f'local.get ${n}','i16x8')+[f'local.set ${name}']
     for m,(_,name)in enumerate(leaves):out+=[f'local.get ${name}',f'local.set $w{m}_{j}_{k}']
   for m in range(49):
    if j==0:out.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qoff)))')
    for k in range(32):
     out.append(f'(local.tee $x{m}_{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 else f'local.get $x{m}_{k}')
     out+=[f'local.get $w{m}_{j}_{k}','i32x4.dot_i16x8_s']
     if k:out.append('i32x4.add')
    out.append(f'local.set $pc{m}')
   out+=rec
   for ti in range(4):
    out+=[f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $output_bytes))))']
    for ri in range(4):out+=[f'local.get $pc{regs[ti][ri]}',f'local.set $c{ri}']
    for i,(x,y,mask)in enumerate([(0,1,[0,4,1,5]),(2,3,[0,4,1,5]),(0,1,[2,6,3,7]),(2,3,[2,6,3,7])]):out+=[f'local.get $c{x}',f'local.get $c{y}',shuffle(mask),f'local.set $u{i}']
    for part,(x,y,mask)in enumerate([(0,1,[0,1,4,5]),(0,1,[2,3,6,7]),(2,3,[0,1,4,5]),(2,3,[2,3,6,7])]):
     offset=j*64+part*16
     out+=['local.get $yp']+(['v128.const i32x4 0 0 0 0']if seed else ['local.get $yp',f'v128.load offset={offset}'])+[f'local.get $u{x}',f'local.get $u{y}',shuffle(mask),'f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{j*4+part}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
    out+=['))']
  return out
 lines+=['(block $done','(br_if $done(i32.eqz(local.get $n)))']+row(True)+['(local.set $t(i32.const 4))','(loop $tokens','(br_if $done(i32.ge_u(local.get $t)(local.get $n)))']+row(False)+['(local.set $t(i32.add(local.get $t)(i32.const 4)))','br $tokens','))','))']
 text='\n'.join(lines)+'\n';assert text.count('(')==text.count(')');count=text.count('(local $');assert count<10000,(tile,count)
 return text,count,symbol

def main():
 d=ROOT/'artifacts/s2-k2-kernels-v1';d.mkdir(exist_ok=False)
 source=ROOT/'artifacts/s2-winograd-hoisted-v1/frozen-plan.py';(d/'plan.py').write_bytes(source.read_bytes())
 ns=dict(__name__='rank49',__file__=str(d/'plan.py'));exec(compile(source.read_text(),str(source),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();bounds=[]
 for name,e in c.symbols.items():
  terms={}
  for m,sign in e.items():
   for ai,av in a.symbols[leaves[m][0]].items():
    for bi,bv in b.symbols[leaves[m][1]].items():terms[ai,bi]=terms.get((ai,bi),0)+sign*av*bv
  bound=sum(abs(v)for v in terms.values())*127*128*64;assert bound<2**31;bounds.append(dict(name=name,bound=bound))
 kernels=[]
 for tile in [80,16]:
  for seed in [False,True]:
   text,count,symbol=kernel(ns,tile,seed);p=d/(str(tile)+('_seed'if seed else '')+'.wat');p.write_text(text);kernels.append(dict(path=str(p.relative_to(ROOT)),symbol=symbol,locals=count))
 files=[Path(__file__),source,d/'plan.py']+[ROOT/k['path']for k in kernels]
 r=dict(rank=49,kernels=kernels,bounds=bounds,max_i32_bound=max(x['bound']for x in bounds),input_i16_bound=max(sum(abs(v)for v in e.values())for e in a.symbols.values())*127,weight_i16_bound=max(sum(abs(v)for v in e.values())for e in b.symbols.values())*128,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},wasm_validated=False,scope=__doc__)
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(kernels=kernels,max_i32_bound=r['max_i32_bound'])))
if __name__=='__main__':main()
