#!/usr/bin/env python3
"""Generate exact bounded full-runtime K256 SIMD bodies without LLVM expansion."""
from pathlib import Path
import argparse
ap=argparse.ArgumentParser();ap.add_argument("--layout",choices=["loop","guarded48"],default="guarded48");args=ap.parse_args()
ROOT=Path(__file__).resolve().parents[1]
# Validate the exact lane mapping: each integer block dot contains all 256
# original products for exactly one output row. Every partial lane sum is
# bounded by 128*127*128 and the final sum by 256*127*128 < 2^31.
assert 256*127*128 < 2**31
for output_group in range(8):
    lanes=[]
    for half in range(2):
        pair=output_group*2+half
        lanes.extend([[(pair*2+lane//2,g*4+(lane%2)*2+k) for g in range(64) for k in range(2)] for lane in range(4)])
    for out in range(4):
        combined=lanes[out*2]+lanes[out*2+1]
        assert sorted(combined)==[(output_group*4+out,k) for k in range(256)]
lines=['(module (func (export "__imajev_pair_accumulate") (param $q i32) (param $w i32) (param $cols i32) (param $start i32) (param $sx i32) (param $stride i32) (param $sw i32) (param $sums i32) (param $n i32)']
lines+=['(local $base i32) (local $t i32) (local $qp i32) (local $sp i32) (local $wp i32) (local $yp i32) (local $xscale v128) (local $a v128) (local $b v128)']
for j in range(16):
 for g in range(64):lines.append(f'(local $w{j}_{g} v128)')
for g in range(64):lines.append(f'(local $x{g} v128)')
for j in range(8):lines.append(f'(local $scale{j} v128)')
# Input has already been duplicated as exact I16 [qK4,qK4].
for j in range(8):lines += [f'(local.set $scale{j} (v128.load offset={j*16} (local.get $sw)))']
for j in range(16):
 lines.append(f'(local.set $wp (i32.add (local.get $w) (i32.add (i32.mul (local.get $cols) (i32.const {j*2})) (i32.shl (local.get $start) (i32.const 1)))))')
 for g in range(64):lines.append(f'(local.set $w{j}_{g} (i16x8.extend_low_i8x16_u (v128.load64_zero offset={g*8} (local.get $wp))))'.replace('extend_low_i8x16_u','extend_low_i8x16_s'))
row=[
 '(local.set $qp (i32.add (local.get $q) (i32.shl (i32.add (i32.mul (local.get $t) (local.get $cols)) (local.get $start)) (i32.const 2))))',
 '(local.set $yp (i32.add (local.get $sums) (i32.shl (local.get $t) (i32.const 7))))',
 '(local.set $xscale (f32x4.splat (f32.load (i32.add (local.get $sx) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2))))))']
for g in range(64):row.append(f'(local.set $x{g} (v128.load offset={g*16} (local.get $qp)))')
for j in range(8):
 for half in range(2):
  pair=j*2+half
  # Instruction-form keeps each partial sum on the operand stack. No source
  # optimizer can reassociate the integer sum before this body is injected.
  def dot(lo,hi):
   if hi-lo==1:return [f'local.get $x{lo}',f'local.get $w{pair}_{lo}','i32x4.dot_i16x8_s']
   mid=(lo+hi)//2
   return dot(lo,mid)+dot(mid,hi)+['i32x4.add']
  row+=dot(0,64)
  row.append(f'local.set ${"a" if half==0 else "b"}')
 row+=['local.get $yp',f'v128.load offset={j*16}','local.get $a','local.get $b','i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27','local.get $a','local.get $b','i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31','i32x4.add','f32x4.convert_i32x4_s','local.get $xscale','f32x4.mul',f'local.get $scale{j}','f32x4.mul','f32x4.add',f'local.set $a', 'local.get $yp','local.get $a',f'v128.store offset={j*16}']
if args.layout=='loop':
 lines+=['(local.set $t (i32.const 0))','(block $done (loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+row+['(local.set $t (i32.add (local.get $t) (i32.const 1)))','br $tokens','))']
else:
 lines+=['(if (i32.gt_u (local.get $n) (i32.const 132)) (then unreachable))','(local.set $base (i32.const 0))','(block $done (loop $chunks','(br_if $done (i32.ge_u (local.get $base) (local.get $n)))']
 for t in range(48):lines+=[f'(if (i32.gt_u (local.get $n) (i32.add (local.get $base) (i32.const {t}))) (then',f'(local.set $t (i32.add (local.get $base) (i32.const {t})))']+row+['))']
 lines+=['(local.set $base (i32.add (local.get $base) (i32.const 48)))','br $chunks','))']
lines+=['))']
p=ROOT/f'artifacts/prefix_codec/full-wat/{args.layout}.wat';p.parent.mkdir(parents=True,exist_ok=True);s='\n'.join(lines)+'\n'
if not p.exists() or p.read_text()!=s:p.write_text(s)
