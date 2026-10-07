#!/usr/bin/env python3
"""Rank343 K2: one 32-output tile, four independent outputs per SIMD lane.

ABI: q points to 343 leaf pointers. Each leaf contains [group8][block256]
[32 I16]. w points to a single prepared tile pointer with [block][leaf343]
[16 K pairs][four output lanes][two I16]. sx already points to the current
block scale for token0; stride is the full output row stride. No hidden work.
"""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'artifacts/s3-k2-prepared-kernels-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shuffle(mask):return 'i8x16.shuffle '+' '.join(str(i*4+b)for i in mask for b in range(4))
def kernel(ns,seed):
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,capacity=ns['reconstruct'](c,roots)
 symbol='__imajev_s3_k2_32'+('_seed'if seed else '')
 lines=[f'(module(func(export "{symbol}")'+''.join(f'(param ${p} i32)'for p in ['q','w','cols','start','sx','stride','sw','sums','n']),
 '(local $t i32)(local $qp i32)(local $yp i32)(local $qoff i32)(local $bp i32)(local $input_stride i32)(local $output_bytes i32)']
 for i in range(capacity):lines.append(f'(local $pc{i} v128)')
 for m in range(343):
  for k in range(16):lines.append(f'(local $w{m}_{k} v128)')
 for i in range(8):lines.append(f'(local $sx{i} v128)(local $sw{i} v128)')
 lines+=['(local.set $input_stride(i32.shr_u(local.get $cols)(i32.const 8)))','(local.set $output_bytes(i32.shl(local.get $stride)(i32.const 2)))',
 '(local.set $bp(i32.add(i32.load(local.get $w))(i32.mul(i32.shr_u(local.get $start)(i32.const 8))(i32.const 87808))))']
 for i in range(8):lines.append(f'(local.set $sw{i}(v128.load offset={i*16}(local.get $sw)))')
 def row(first):
  out=['(local.set $qoff(i32.add(i32.shr_u(i32.mul(local.get $t)(local.get $cols))(i32.const 5))(i32.shr_u(local.get $start)(i32.const 2))))']
  for ti in range(8):out.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then(local.set $sx{ti}(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $input_stride))(i32.const 2)))))))')
  for m in range(343):
   out.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qoff)))')
   for k in range(16):
    out.append(f'(v128.load32_splat offset={k*4}(local.get $qp))')
    out.append(f'(local.tee $w{m}_{k}(v128.load offset={m*256+k*16}(local.get $bp)))'if first else f'local.get $w{m}_{k}')
    out.append('i32x4.dot_i16x8_s')
    if k:out.append('i32x4.add')
   out.append(f'local.set $pc{m}')
  out+=rec
  for ti in range(8):
   out+=[f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti}))(local.get $n))(then',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {ti}))(local.get $output_bytes))))']
   for lane in range(4):
    for base in [0,4]:
     offset=(lane*8+base)*4
     out+=['local.get $yp']+(['v128.const i32x4 0 0 0 0']if seed else ['local.get $yp',f'v128.load offset={offset}'])
     mask=[0,4,1,5]if lane<2 else [2,6,3,7]
     for pair in [0,2]:out +=[f'local.get $pc{regs[ti][base+pair]}',f'local.get $pc{regs[ti][base+pair+1]}',shuffle(mask)]
     out +=[shuffle([0,1,4,5]if lane%2==0 else [2,3,6,7]),'f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{lane*2+base//4}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
   out+=['))']
  return out
 lines+=['(block $done','(br_if $done(i32.eqz(local.get $n)))']+row(True)+['(local.set $t(i32.const 8))','(loop $tokens','(br_if $done(i32.ge_u(local.get $t)(local.get $n)))']+row(False)+['(local.set $t(i32.add(local.get $t)(i32.const 8)))','br $tokens','))','))']
 text='\n'.join(lines)+'\n';assert text.count('(')==text.count(')');count=text.count('(local $');assert count<10000
 return text,count,symbol
def main():
 d=DIRECTORY;d.mkdir(exist_ok=False)
 source=ROOT/'artifacts/s3-winograd-v1/frozen-generator.py';(d/'plan.py').write_bytes(source.read_bytes())
 ns={'__name__':'plan','__file__':str(d/'plan.py')};exec(compile(source.read_text(),str(source),'exec'),ns)
 a,b,c,l,roots,bound=ns['plan']();kernels=[]
 for seed in [False,True]:
  text,count,symbol=kernel(ns,seed);p=d/('32_seed.wat'if seed else '32.wat');p.write_text(text);kernels.append(dict(path=str(p.relative_to(ROOT)),symbol=symbol,locals=count))
 files=[Path(__file__),source,d/'plan.py']+[ROOT/k['path']for k in kernels]
 report=dict(rank=343,kernels=kernels,weight_vectors=5488,reconstruction_nodes=len(c.nodes),input_i16_bound=max(sum(abs(v)for v in e.values())for e in a.symbols.values())*127,weight_i16_bound=max(sum(abs(v)for v in e.values())for e in b.symbols.values())*128,unreduced_bound=bound,final_dot_bound=256*127*128,prepared_block_bytes=87808,prepared_weight_ratio=10.71875,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['source_hashes','scope']}))
if __name__=='__main__':main()
