#!/usr/bin/env python3
"""Keep each F32 cell accumulator on the operand stack, preserving column order."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kernel():
    width = 128
    vectors = width // 4
    lines = ['(module (func (export "__imajev_f32_wide") ' + ' '.join(
        f'(param ${p} i32)' for p in ['q', 'w', 'cols', 'start', 'sx', 'stride', 'sw', 'sums', 'n']),
        '(local $t i32) (local $qp i32) (local $yp i32)']
    lines += [f'(local $wp{p} i32)' for p in range(width // 32)]
    lines += [f'(local $x{c} v128)' for c in range(64)]
    lines += [f'(local $w{c}_{r} v128)' for c in range(64) for r in range(vectors)]

    def token(first):
        out = []
        if first:
            out += [f'(local.set $wp{p} (i32.add (local.get $w) (i32.shl (i32.add (local.get $start) (i32.mul (local.get $cols) (i32.const {p}))) (i32.const 7))))' for p in range(width // 32)]
        out += ['(local.set $qp (i32.add (local.get $q) (i32.shl (i32.add (i32.mul (local.get $t) (local.get $cols)) (local.get $start)) (i32.const 2))))',
                '(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2))))']
        out += [f'(local.set $x{c} (v128.load32_splat offset={c * 4} (local.get $qp)))' for c in range(64)]
        for r in range(vectors):
            # Address below the accumulator survives all ascending-column
            # mul/add operations and is consumed by the final store.
            out += ['local.get $yp', 'local.get $yp', f'v128.load offset={r * 16}']
            for c in range(64):
                out += [f'local.get $x{c}']
                out += [f'(local.tee $w{c}_{r} (v128.load offset={c * 128 + (r % 8) * 16} (local.get $wp{r // 8})))' if first else f'local.get $w{c}_{r}']
                out += ['f32x4.mul', 'f32x4.add']
            out += [f'v128.store offset={r * 16}']
        return out

    lines += ['(local.set $t (i32.const 0))', '(block $done (br_if $done (i32.eqz (local.get $n)))'] + token(True)
    lines += ['(local.set $t (i32.const 1))', '(loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))'] + token(False)
    lines += ['(local.set $t (i32.add (local.get $t) (i32.const 1)))', 'br $tokens', '))', '))']
    return '\n'.join(lines) + '\n'


def main():
    p = ROOT / 'scripts/build_f32_output128_probe.py'
    source = p.read_text().replace('artifacts/f32-output128-v1/build', 'artifacts/f32-stack-v1/build')
    namespace = dict(__file__=__file__, __name__='f32_stack_builder')
    exec(compile(source, str(p), 'exec'), namespace)
    namespace['kernel'] = kernel
    namespace['main']()


if __name__ == '__main__':
    main()
