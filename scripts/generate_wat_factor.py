#!/usr/bin/env python3
"""Exact pair-factor dots; fixed weights and per-token operands loaded once."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
lines=['(module (func (export "__imajev_pair_accumulate")',
       '(param $q i32) (param $w i32) (param $cols i32) (param $start i32)',
       '(param $sx i32) (param $stride i32) (param $sw i32) (param $sums i32) (param $n i32)',
       '(local $t i32) (local $qp i32) (local $wp i32) (local $yp i32)',
       '(local $xscale v128) (local $xfactor v128)',
       '(local $a v128) (local $b v128) (local $c v128) (local $d v128)',
       '(local $ab v128) (local $cd v128)']
for row in range(32):
    for g in range(32):lines.append(f'(local $w{row}_{g} v128)')
for g in range(32):lines.append(f'(local $x{g} v128)')
for j in range(8):lines += [f'(local $scale{j} v128)',f'(local $factor{j} v128)']


def row(first):
    out=['(local.set $qp (i32.add (local.get $q) (i32.add'
         ' (i32.mul (local.get $t) (i32.mul (local.get $stride) (i32.const 516)))'
         ' (i32.mul (i32.shr_u (local.get $start) (i32.const 8)) (i32.const 516)))))',
         '(local.set $yp (i32.add (local.get $sums) (i32.shl (local.get $t) (i32.const 7))))',
         '(local.set $xfactor (v128.load32_splat offset=512 (local.get $qp)))']
    for j in range(8):
        out += ['local.get $yp','local.get $yp',f'v128.load offset={j*16}']
        for lane in range(4):
            r=j*4+lane
            if first:
                out.append('(local.set $wp (i32.add (local.get $w) (i32.add'
                           f' (i32.mul (local.get $stride) (i32.const {r*260}))'
                           ' (i32.mul (i32.shr_u (local.get $start) (i32.const 8)) (i32.const 260)))))')
            def dot(lo,hi):
                if hi-lo==1:
                    ops=[]
                    for qi,wi in [(lo,lo+16),(lo+16,lo)]:
                        ops += ([f'(local.tee $x{qi} (v128.load offset={qi*16} (local.get $qp)))'] if r==0 else [f'local.get $x{qi}'])
                        ops += ([f'(local.tee $w{r}_{wi} (v128.load8x8_s offset={wi*8} (local.get $wp)))'] if first else [f'local.get $w{r}_{wi}'])
                        ops.append('i16x8.add')
                    return ops+['i32x4.dot_i16x8_s']
                mid=(lo+hi)//2
                return dot(lo,mid)+dot(mid,hi)+['i32x4.add']
            out += dot(0,16)+[f'local.set ${"abcd"[lane]}']
        for name, left,right in [('ab','a','b'),('cd','c','d')]:
            out += [f'local.get ${left}',f'local.get ${right}',
                    'i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23',
                    f'local.get ${left}',f'local.get ${right}',
                    'i8x16.shuffle 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31',
                    'i32x4.add',f'local.set ${name}']
        out += ['local.get $ab','local.get $cd',
                'i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27',
                'local.get $ab','local.get $cd',
                'i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31',
                'i32x4.add']
        if first:
            out += ['i32.const 0','i32x4.splat']
            for lane in range(4):
                r=4*j+lane
                out += ['(i32.load (i32.add (local.get $w) (i32.add'
                        f' (i32.mul (local.get $stride) (i32.const {r*260}))'
                        ' (i32.add (i32.mul (i32.shr_u (local.get $start) (i32.const 8)) (i32.const 260)) (i32.const 256)))))',
                        f'i32x4.replace_lane {lane}']
            out.append(f'local.tee $factor{j}')
        else:out.append(f'local.get $factor{j}')
        out += ['i32x4.sub','local.get $xfactor','i32x4.sub','f32x4.convert_i32x4_s']
        out.append('(local.tee $xscale (v128.load32_splat (i32.add (local.get $sx) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2)))))' if j==0 else 'local.get $xscale')
        out += ['f32x4.mul', f'(local.tee $scale{j} (v128.load offset={j*16} (local.get $sw)))' if first else f'local.get $scale{j}',
                'f32x4.mul','f32x4.add',f'v128.store offset={j*16}']
    return out


lines += ['(if (i32.gt_u (local.get $n) (i32.const 132)) (then unreachable))',
          '(block $done','(br_if $done (i32.eqz (local.get $n)))','(local.set $t (i32.const 0))']
lines += row(True)+['(local.set $t (i32.const 1))','(loop $tokens','(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
lines += row(False)+['(local.set $t (i32.add (local.get $t) (i32.const 1)))','br $tokens',')))',')']
path=ROOT/'artifacts/prefix_codec/factor-build/factor.wat'
path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n')
