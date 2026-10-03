#!/usr/bin/env python3
"""Generate the current C64 leaf from the historical exact C4 packed reduction."""
from pathlib import Path
import argparse
ap=argparse.ArgumentParser();ap.add_argument("--columns",type=int,choices=[64],default=64);C=ap.parse_args().columns
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / 'crates/imajev-runtime/src/strassen_prepacked.rs').read_text()
marker = '#[cfg(all(target_arch = "wasm32", feature = "experimental-strassen-packed-reduction"))]'
a = src.index('unsafe fn dot_tile<const R: usize>', src.index(marker))
b = src.index('\n#[cfg(test)]', a)
leaf = src[a:b].replace('StrassenWeight', 'i16')
for old, new in [('[[i32; 4]; R]', f'[[i32; {C}]; R]'),
                 ('[*const i16; 4]', f'[*const i16; {C}]'),
                 ('[[v128; 16]; 4]', f'[[v128; 16]; {C}]'),
                 ('[[0i32; 4]; R]', f'[[0i32; {C}]; R]')]:
    leaf = leaf.replace(old, new)
weights = '[' + ','.join('[' + ','.join(
    f'v128_load(wp[{j}].add({k * 8}).cast())' for k in range(16)) + ']'
    for j in range(C)) + ']'
leaf = leaf.replace('core::array::from_fn(|j| core::array::from_fn(|k| v128_load(wp[j].add(k * 8).cast())))', weights)
inputs = '[' + ','.join(f'v128_load(qp[$i].add({k * 8}).cast())' for k in range(16)) + ']'
leaf = leaf.replace('core::array::from_fn(|k| v128_load(qp[$i].add(k * 8).cast()))', inputs)
a = leaf.index('        let a0 = dot!')
b = leaf.index('\n    } }; }', a)
code = ''
for group in range(C//4):
    j = group * 4
    code += f'''        {{let a0=dot!(input,{j});let a1=dot!(input,{j+1});let a2=dot!(input,{j+2});let a3=dot!(input,{j+3});
        let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
        let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
        let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
        v128_store(out[$i].as_mut_ptr().add({j}).cast(),total);}}\n'''
leaf = leaf[:a] + code.rstrip() + leaf[b:]
# Private callers validate the complete operand spans and dispatch bounds.
# These indices cannot wrap; avoid repeating overflow branches in the leaf.
leaf=leaf.replace('i * stride','i.wrapping_mul(stride)').replace('j * stride','j.wrapping_mul(stride)')
(ROOT / 'scripts/strassen16_bench/src/kernel.rs').write_text(
    f'//! Generated explicit I16 Strassen leaf: {C} outputs share each input load.\n'
    '#[target_feature(enable="simd128")]\npub(crate) ' + leaf)
