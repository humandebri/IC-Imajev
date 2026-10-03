#!/usr/bin/env python3
"""Share fixed expansion across all real rows; compare exact integer sum trees."""
from pathlib import Path
from functools import reduce
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'crates/imajev-runtime/src/output_pairs_simd.rs').read_text()
s=s.replace('scripts/generate_output_pairs.py','scripts/generate_linear_pair.py')
s=s.replace('accumulate<const R:usize>','accumulate<const LINEAR:bool>').replace('sums:&mut[[f32;32];R]','sums:&mut[[f32;32]]').replace('const {assert!(R<=48);}','debug_assert!(sums.len()<=132);')
a=s.index('macro_rules! dot ');b=s.index('macro_rules! group ',a);balanced=s[a:b].split('=>{',1)[1].rsplit('};}',1)[0]
terms=[f'i32x4_dot_i16x8($input[{g}],weights[$j][{g}])' for g in range(64)]
linear=reduce(lambda a,b:f'i32x4_add({a},{b})',terms)
s=s[:a]+'macro_rules! dot {($input:ident,$j:expr)=>{{if LINEAR {'+linear+'}else{'+balanced+'}}};}\n'+s[b:]
s=s.replace('($i:literal,$input:ident,$j:literal)','($i:expr,$input:ident,$j:literal)')
s=s.replace('macro_rules! rows {($($i:literal),*)=>{$(if R>$i {','for i in 0..sums.len() {')
s=s.replace('(($i as usize).wrapping_mul(cols)','(i.wrapping_mul(cols)').replace('group!($i,input,','group!(i,input,')
a=s.index('})*};}\nrows!(');b=s.index('\n}',a);s=s[:a]+'}\n'+s[b:]
p=ROOT/'scripts/linear_pair_bench/src/kernel.rs'
if not p.exists() or p.read_text()!=s:p.write_text(s)

static=(ROOT/'crates/imajev-runtime/src/output_pairs_simd.rs').read_text().replace('scripts/generate_output_pairs.py','scripts/generate_linear_pair.py').replace('assert!(R<=48)','assert!(R<=87)')
a=static.index('rows!(');b=static.index(');',a);static=static[:a]+'rows!('+','.join(map(str,range(87)))+static[b:]
p=ROOT/'scripts/linear_pair_bench/src/kernel_static.rs'
if not p.exists() or p.read_text()!=static:p.write_text(static)
