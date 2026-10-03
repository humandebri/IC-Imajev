#!/usr/bin/env python3
"""Exact pair dot: initialize read-only operands on first use, then reuse them.

Weight expansion happens once for all real token rows. Input vectors are loaded
once per row, in their first dot. Integer trees and floating-point order match
the adopted pair kernel; no new quantization or persistent query state.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = ['(module (func (export "__imajev_pair_accumulate")'
         ' (param $q i32) (param $w i32) (param $cols i32)'
         ' (param $start i32) (param $sx i32) (param $stride i32)'
         ' (param $sw i32) (param $sums i32) (param $n i32)',
         '(local $t i32) (local $qp i32) (local $wp i32)'
         ' (local $yp i32) (local $xscale v128)'
         ' (local $a v128) (local $b v128)']
for pair in range(16):
    for g in range(64):
        lines.append(f'(local $w{pair}_{g} v128)')
for g in range(64):
    lines.append(f'(local $x{g} v128)')
for j in range(8):
    lines.append(f'(local $scale{j} v128)')


def row(first):
    out = [
        '(local.set $qp (i32.add (local.get $q) (i32.shl'
        ' (i32.add (i32.mul (local.get $t) (local.get $cols))'
        ' (local.get $start)) (i32.const 2))))',
        '(local.set $yp (i32.add (local.get $sums)'
        ' (i32.shl (local.get $t) (i32.const 7))))',
    ]
    for j in range(8):
        # Keep the destination address and prior F32 sum on the operand stack.
        out += ['local.get $yp', 'local.get $yp', f'v128.load offset={j*16}']
        for half in range(2):
            pair = 2*j + half
            if first:
                out.append('(local.set $wp (i32.add (local.get $w)'
                           f' (i32.add (i32.mul (local.get $cols) (i32.const {pair*2}))'
                           ' (i32.shl (local.get $start) (i32.const 1)))))')

            def dot(lo, hi):
                if hi - lo == 1:
                    ops = ([f'(local.tee $x{lo} (v128.load offset={lo*16}'
                            ' (local.get $qp)))'] if pair == 0
                           else [f'local.get $x{lo}'])
                    ops += ([f'(local.tee $w{pair}_{lo}'
                             f' (v128.load8x8_s offset={lo*8} (local.get $wp)))']
                            if first else [f'local.get $w{pair}_{lo}'])
                    return ops + ['i32x4.dot_i16x8_s']
                mid = (lo + hi)//2
                return dot(lo, mid) + dot(mid, hi) + ['i32x4.add']

            out += dot(0, 64)
            out.append(f'local.tee ${"a" if half == 0 else "b"}')
        out += [
            'i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27',
            'local.get $a', 'local.get $b',
            'i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31',
            'i32x4.add', 'f32x4.convert_i32x4_s',
        ]
        if j == 0:
            out.append('(local.tee $xscale (v128.load32_splat (i32.add'
                       ' (local.get $sx) (i32.shl (i32.mul (local.get $t)'
                       ' (local.get $stride)) (i32.const 2)))))')
        else:
            out.append('local.get $xscale')
        out.append('f32x4.mul')
        out.append(f'(local.tee $scale{j} (v128.load offset={j*16}'
                   ' (local.get $sw)))' if first else f'local.get $scale{j}')
        out += ['f32x4.mul', 'f32x4.add', f'v128.store offset={j*16}']
    return out


lines += ['(if (i32.gt_u (local.get $n) (i32.const 132)) (then unreachable))',
          '(block $done', '(br_if $done (i32.eqz (local.get $n)))',
          '(local.set $t (i32.const 0))']
lines += row(True)
lines += ['(local.set $t (i32.const 1))', '(loop $tokens',
          '(br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
lines += row(False)
lines += ['(local.set $t (i32.add (local.get $t) (i32.const 1)))',
          'br $tokens', ')))', ')']
path = ROOT/'artifacts/prefix_codec/full-wat/reuse.wat'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('\n'.join(lines)+'\n')
