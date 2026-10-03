#!/usr/bin/env python3
"""Exact standard-SIMD two-token/four-output Strassen1 kernel, no lossy transform."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
A=[{0:1,3:1},{2:1,3:1},{0:1},{3:1},{0:1,1:1},{2:1,0:-1},{1:1,3:-1}]
B=[{0:1,3:1},{0:1},{1:1,3:-1},{2:1,0:-1},{3:1},{0:1,1:1},{2:1,3:1}]
C=[{0:1,3:1,4:-1,6:1},{2:1,4:1},{1:1,3:1},{0:1,1:-1,2:1,5:1}]
for out,c in enumerate(C):
 p={}
 for m,v in c.items():
  for ai,av in A[m].items():
   for bi,bv in B[m].items():p[ai,bi]=p.get((ai,bi),0)+v*av*bv
 p={k:v for k,v in p.items() if v}
 assert p=={(out//2*2+k,k*2+out%2):1 for k in range(2)}
 assert sum(abs(v)*sum(abs(x) for x in A[m].values())*127*sum(abs(x) for x in B[m].values())*128*128 for m,v in c.items())<2**31
lines=['(module (func (export "__imajev_pair_accumulate") (param $q i32) (param $w i32) (param $cols i32) (param $start i32) (param $sx i32) (param $stride i32) (param $sw i32) (param $sums i32) (param $n i32)', '(local $t i32) (local $keep i32) (local $qp i32) (local $wp i32) (local $yp i32) (local $a v128) (local $value v128) (local $sx0 v128) (local $sx1 v128)']
for m in range(7):
 lines.append(f'(local $wp{m} i32) (local $p{m} v128)')
 for g in range(32):lines.append(f'(local $x{m}_{g} v128)')
 for j in range(8):
  for g in range(32):lines.append(f'(local $w{m}_{j}_{g} v128)')
for j in range(8):lines.append(f'(local $sw{j} v128)')
for j in range(8):lines.append(f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))')
for m in range(7):
 lines.append(f'(local.set $wp{m} (i32.add (i32.load offset={m*4} (local.get $w)) (i32.shl (local.get $start) (i32.const 1))))')
 for j in range(8):
  lines.append(f'(local.set $wp (i32.add (local.get $wp{m}) (i32.mul (local.get $cols) (i32.const {j*2}))))')
  for g in range(32):lines.append(f'(local.set $w{m}_{j}_{g} (v128.load offset={g*16} (local.get $wp)))')
lines+=['(local.set $t (i32.const 0))','(block $done (loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))','(local.set $keep (i32.lt_u (i32.add (local.get $t) (i32.const 1)) (local.get $n)))','(local.set $yp (i32.add (local.get $sums) (i32.shl (local.get $t) (i32.const 7))))','(local.set $sx0 (f32x4.splat (f32.load (i32.add (local.get $sx) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2))))))','(if (local.get $keep) (then (local.set $sx1 (f32x4.splat (f32.load (i32.add (local.get $sx) (i32.shl (i32.mul (i32.add (local.get $t) (i32.const 1)) (local.get $stride)) (i32.const 2))))))))']
for m in range(7):
 lines.append(f'(local.set $qp (i32.add (i32.load offset={m*4} (local.get $q)) (i32.add (i32.mul (local.get $t) (local.get $cols)) (i32.shl (local.get $start) (i32.const 1)))))')
 for g in range(32):lines.append(f'(local.set $x{m}_{g} (v128.load offset={g*16} (local.get $qp)))')

def combine(terms):
 keys=list(terms);v=keys[0];assert terms[v]==1
 result=[f'local.get $p{v}']
 for v in keys[1:]:result += [f'local.get $p{v}','i32x4.add' if terms[v]==1 else 'i32x4.sub']
 return result
for j in range(8):
 for m in range(7):
  for g in range(32):
   lines += [f'local.get $x{m}_{g}',f'local.get $w{m}_{j}_{g}','i32x4.dot_i16x8_s']
   if g:lines.append('i32x4.add')
  lines+=['local.tee $a','local.get $a','local.get $a','i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11','i32x4.add',f'local.set $p{m}']
 for t in range(2):
  if t:lines+=['(if (local.get $keep) (then']
  lines+=['local.get $yp',f'v128.load offset={t*128+j*16}']+combine(C[t*2])+combine(C[t*2+1])+['i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27','f32x4.convert_i32x4_s',f'local.get $sx{t}','f32x4.mul',f'local.get $sw{j}','f32x4.mul','f32x4.add','local.set $value','local.get $yp','local.get $value',f'v128.store offset={t*128+j*16}']
  if t:lines+=['))']
lines+=['(local.set $t (i32.add (local.get $t) (i32.const 2)))','br $tokens','))','))']
p=ROOT/'scripts/wat_s1_bench/kernel.wat';s='\n'.join(lines)+'\n'
if not p.exists() or p.read_text()!=s:p.write_text(s)
print('Symbolic identity and INT32 bound verified; 2 tokens x 4 original outputs, 7 K128 products shared')
