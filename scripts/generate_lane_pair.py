#!/usr/bin/env python3
"""Generate the fixed R44/C32 SIMD tile. No model weights are embedded."""
from pathlib import Path

def tree(values):
    if len(values)==1:return values[0]
    m=len(values)//2
    return f'i32x4_add({tree(values[:m])},{tree(values[m:])})'

p=Path(__file__).resolve().parent/'lane_pair_bench/src/kernel.rs'
s='''//! Deterministic source from scripts/generate_lane_pair.py.
//! Q and each weight pair occupy eight lanes per K4 group.
//! Integer associativity is exact: |block sum| <= 256*127*128 < 2^31.
#[target_feature(enable="simd128")]
pub(crate) unsafe fn accumulate(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:&mut[[f32;32];44]) {
use core::arch::wasm32::*;
let ws:[v128;8]=core::array::from_fn(|j|v128_load(sw.add(j*4).cast()));
let wp:[*const i8;16]=core::array::from_fn(|j|w.add(j*cols*2+start*2));
let mut weights=[[i32x4_splat(0);64];16];
macro_rules! pair {($j:literal)=>{weights[$j]=[
'''
s+=',\n'.join(f'i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add({g*8}).cast()))' for g in range(64))
s+='\n];};}\npair!(0);pair!(1);pair!(2);pair!(3);pair!(4);pair!(5);pair!(6);pair!(7);pair!(8);pair!(9);pair!(10);pair!(11);pair!(12);pair!(13);pair!(14);pair!(15);\n'
s+='macro_rules! dot {($input:ident,$j:expr)=>{'+tree([f'i32x4_dot_i16x8($input[{g}],weights[$j][{g}])' for g in range(64)])+'};}\n'
s+='''macro_rules! group {($i:literal,$input:ident,$j:literal)=>{{
let a=dot!($input,$j/2);let b=dot!($input,$j/2+1);
let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
let scaled=f32x4_mul(f32x4_mul(f32x4_convert_i32x4(total),f32x4_splat(*sx.add($i*stride))),ws[$j/4]);
let dst=sums[$i].as_mut_ptr().add($j);v128_store(dst.cast(),f32x4_add(v128_load(dst.cast()),scaled));
}};}
macro_rules! rows {($($i:literal),*)=>{$({
let input:[v128;64]=core::array::from_fn(|g|v128_load(q.add($i*cols*2+start*2+g*8).cast()));
group!($i,input,0);group!($i,input,4);group!($i,input,8);group!($i,input,12);group!($i,input,16);group!($i,input,20);group!($i,input,24);group!($i,input,28);
})*};}
'''
s+='rows!('+','.join(map(str,range(44)))+');\n}\n'
# LLVM retains a 64-element from_fn loop per token. Explicit bounded loads
# remove query-local array construction loops without changing any value.
s=s.replace('let input:[v128;64]=', 'let row=q.add(($i*cols+start)*2);\nlet input:[v128;64]=')
s=s.replace('core::array::from_fn(|g|v128_load(q.add($i*cols*2+start*2+g*8).cast()))',
            '['+','.join(f'v128_load(row.add({g*8}).cast())' for g in range(64))+']')
p.write_text(s)
