#!/usr/bin/env python3
"""Generate larger exact Delta SIMD groups for local instruction experiments."""
import argparse
import pathlib


def generate(group, fallback):
    width = group * 4
    before = fallback.index('        for start in (0..dv).step_by(16) {')
    tail = fallback.index('            } else {', before)
    end = fallback.index('\n    for i in 0..dk {', tail)
    # Keep the established four-lane remainder path and transpose order.
    remainder = fallback[tail:end]
    body = [f'        for start in (0..dv).step_by({width}) {{',
            f'            if start + {width} <= dv {{']
    body += [f'                let mut mem{j} = f32x4_splat(0.);' for j in range(group)]
    body += ['                for i in 0..dk {', '                    let p = sp.add(i * dv + start);',
             '                    let ki = f32x4_splat(*k.get_unchecked(t * dk + i));']
    body += [f'                    mem{j} = f32x4_add(mem{j}, f32x4_mul(f32x4_mul(v128_load(p.add({j*4}).cast()), decay), ki));' for j in range(group)]
    body += ['                }']
    for j in range(group):
        body += [f'                let update{j} = f32x4_mul(f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + {j*4}).cast()), mem{j}), b);',
                 f'                let mut o{j} = f32x4_splat(0.);']
    body += ['                for i in 0..dk {', '                    let p = sp.add(i * dv + start);',
             '                    let ki = f32x4_splat(*k.get_unchecked(t * dk + i));',
             '                    let qi = f32x4_splat(*q.get_unchecked(t * dk + i));']
    for j in range(group):
        body += [f'                    let s{j} = f32x4_add(f32x4_mul(v128_load(p.add({j*4}).cast()), decay), f32x4_mul(ki, update{j}));',
                 f'                    v128_store(p.add({j*4}).cast(), s{j});',
                 f'                    o{j} = f32x4_add(o{j}, f32x4_mul(s{j}, qi));']
    body += ['                }']
    body += [f'                v128_store(out.as_mut_ptr().add(t * dv + start + {j*4}).cast(), o{j});' for j in range(group)]
    result = fallback[:before] + '\n'.join(body) + '\n' + remainder + fallback[end:]
    if group > 4:
        # Preserve the established 16-value path for smaller primitive widths.
        result = result.replace('    use core::arch::wasm32::*;',
            f'    if dv < {width} {{ return run_small(q,k,v,g,beta,state,dk,dv); }}\n'
            '    use core::arch::wasm32::*;', 1)
        result += '\n' + fallback.replace('//!', '//', 1).replace('pub unsafe fn run(', 'unsafe fn run_small(', 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', type=int, choices=[4, 8, 16, 32], required=True)
    ap.add_argument('--source', default='artifacts/delta-lanes4/candidate.rs')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate(args.group, pathlib.Path(args.source).read_text()))


if __name__ == '__main__':
    main()
