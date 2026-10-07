#!/usr/bin/env python3
"""Rank343 WWC leaf-outer: batch48 tokens, tile128 outputs, same block256 order.

Prepared weight ABI: w points to four tile32 base addresses. start remains the
K block start. q is343 compact leaf pointers. Each B leaf is cached for all
six token octets; each query leaf is cached for all four output quartets.
"""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kernel(ns,seed):
 a,b,c,leaves,roots,_=ns['plan']();rec,regs,capacity=ns['reconstruct'](c,roots)
 symbol='__imajev_s3_k2_batch48_128'+('_seed'if seed else '')
 lines=[f'(module(func(export "{symbol}")'+''.join(f'(param ${p} i32)'for p in ['q','w','cols','start','sx','stride','sw','sums','n']),
 '(local $t i32)(local $qp i32)(local $yp i32)(local $qoff i32)(local $input_stride i32)(local $output_bytes i32)']
 for i in range(343,capacity):lines.append(f'(local $pc{i} v128)')
 for j in range(4):
  lines.append(f'(local $bp{j} i32)')
  for k in range(16):lines.append(f'(local $w{j}_{k} v128)')
  for ti in range(6):
   for m in range(343):lines.append(f'(local $p{ti}_{j}_{m} v128)')
 for k in range(16):lines.append(f'(local $x{k} v128)')
 for row in range(48):lines.append(f'(local $sx{row} v128)')
 for i in range(32):lines.append(f'(local $sw{i} v128)')
 for i in range(32):lines.append(f'(local.set $sw{i}(v128.load offset={i*16}(local.get $sw)))')
 lines += ['(local.set $input_stride(i32.shr_u(local.get $cols)(i32.const 8)))','(local.set $output_bytes(i32.shl(local.get $stride)(i32.const 2)))']
 for j in range(4):lines.append(f'(local.set $bp{j}(i32.add(i32.load offset={j*4}(local.get $w))(i32.mul(i32.shr_u(local.get $start)(i32.const 8))(i32.const 87808))))')
 lines+=['(block $done','(loop $batch','(br_if $done(i32.ge_u(local.get $t)(local.get $n)))']
 for row in range(48):
  lines.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {row}))(local.get $n))(then(local.set $sx{row}(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const {row}))(local.get $input_stride))(i32.const 2)))))))')
 for m in range(343):
  for j in range(4):
   for k in range(16):lines.append(f'(local.set $w{j}_{k}(v128.load offset={m*256+k*16}(local.get $bp{j})))')
  for ti in range(6):
   lines.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti*8}))(local.get $n))(then')
   lines.append(f'(local.set $qoff(i32.add(i32.shr_u(i32.mul(i32.add(local.get $t)(i32.const {ti*8}))(local.get $cols))(i32.const 5))(i32.shr_u(local.get $start)(i32.const 2))))')
   lines.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qoff)))')
   for j in range(4):
    for k in range(16):
     lines.append(f'(local.tee $x{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 else f'local.get $x{k}')
     lines += [f'local.get $w{j}_{k}','i32x4.dot_i16x8_s']
     if k:lines.append('i32x4.add')
    lines.append(f'local.set $p{ti}_{j}_{m}')
   lines.append('))')
 def reg(i,ti,j):return f'p{ti}_{j}_{i}'if i<343 else f'pc{i}'
 def shuffle(mask):return 'i8x16.shuffle '+' '.join(str(i*4+b)for i in mask for b in range(4))
 for ti in range(6):
  lines.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti*8}))(local.get $n))(then')
  for j in range(4):
   lines += [re.sub(r'\$pc(\d+)',lambda m:'$'+reg(int(m[1]),ti,j),line)for line in rec]
   for row in range(8):
    absolute=ti*8+row
    lines += [f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {absolute}))(local.get $n))(then',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {absolute}))(local.get $output_bytes))))']
    for lane in range(4):
     for base in [0,4]:
      offset=j*128+(lane*8+base)*4
      lines+=['local.get $yp']+(['v128.const i32x4 0 0 0 0']if seed else ['local.get $yp',f'v128.load offset={offset}'])
      mask=[0,4,1,5]if lane<2 else [2,6,3,7]
      for pair in [0,2]:lines +=[f'local.get ${reg(regs[row][base+pair],ti,j)}',f'local.get ${reg(regs[row][base+pair+1],ti,j)}',shuffle(mask)]
      lines +=[shuffle([0,1,4,5]if lane%2==0 else [2,3,6,7]),'f32x4.convert_i32x4_s',f'local.get $sx{absolute}','f32x4.mul',f'local.get $sw{j*8+lane*2+base//4}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
    lines.append('))')
  lines.append('))')
 lines+=['(local.set $t(i32.add(local.get $t)(i32.const 48)))','br $batch','))','))']
 text='\n'.join(lines)+'\n';assert text.count('(')==text.count(')');assert text.count('(local $')<10000
 return text,symbol
def main():
 d=ROOT/'artifacts/s3-k2-batch48-kernels-v1';d.mkdir(exist_ok=False)
 source=ROOT/'artifacts/rank343-mixed-scheme-screen-v1/WWC.py';(d/'plan.py').write_bytes(source.read_bytes())
 ns={'__name__':'plan','__file__':str(d/'plan.py')};exec(compile(source.read_text(),str(source),'exec'),ns)
 items=[];files=[Path(__file__),source,d/'plan.py']
 for seed in [False,True]:
  text,symbol=kernel(ns,seed);p=d/('128_seed.wat'if seed else '128.wat');p.write_text(text);files.append(p)
  items.append(dict(path=str(p.relative_to(ROOT)),symbol=symbol,locals=text.count('(local $'),bytes=p.stat().st_size,seed=seed))
 report=dict(rank=343,scheme='WWC',token_batch=48,output_tile=128,kernels=items,
  query_locals=16,weight_locals=64,leaf_product_locals=8232,
  source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},
  host_execution_verified=False,wasm_execution_verified=False,performance_verified=False,
  scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(items))
if __name__=='__main__':main()
