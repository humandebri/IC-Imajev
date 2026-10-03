#!/usr/bin/env python3
"""Remove per-token/group checked indexing from the exact Strassen tile."""
from pathlib import Path
import argparse
ap=argparse.ArgumentParser();ap.add_argument("--columns",type=int,choices=[64],default=64);C=ap.parse_args().columns
ROOT=Path(__file__).resolve().parents[1]
s='''//! Generated tile. Validated constructors bound every pointer span.
use super::{Prepared,Operands};
use imajev_runtime::int8_kernel::QuantizedRows;
#[target_feature(enable="simd128")]
pub(super) unsafe fn evaluate<const R:usize>(w:&Prepared,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize,t:usize,r:usize,out:&mut[f32]) {
 use core::arch::wasm32::*;
 const{assert!(R<=32);}
 let stride=w.cols/2;let blocks=w.cols/256;let zero=f32x4_splat(0.);
 let mut sums=[[[[zero;4];2];2];R];
 let ap:[*const i16;7]=core::array::from_fn(|m|a.data.as_ptr().add(m*a.pairs*stride+t/2*stride));
 let wp:[*const i16;7]=core::array::from_fn(|m|w.data.as_ptr().add(m*(w.rows/2)*stride+r/2*stride));
 let sp=q.scales().as_ptr().add(t*blocks);
'''
for i in range(C//2):s+=f' let sw{i}=v128_load(sw.as_ptr().add(r+{i*4}).cast());\n'
for g in range(C//4):
    s+=f' let even{g}=i32x4_shuffle::<0,2,4,6>(sw{g*2},sw{g*2+1});let odd{g}=i32x4_shuffle::<1,3,5,7>(sw{g*2},sw{g*2+1});\n'
s+=' for block in 0..blocks {\n  let start=block*128;\n  let products=[\n'
for m in range(7):s+=f'   crate::kernel::dot_tile::<R>(ap[{m}].add(start),wp[{m}].add(start),stride),\n'
s+='''  ];
  macro_rules! row {($i:literal)=>{if R>$i {
   let sx0=f32x4_splat(*sp.add(($i*2)*blocks+block));
   let sx1=f32x4_splat(*sp.add(($i*2+1)*blocks+block));
'''
for g in range(C//4):
    s+='   {\n'
    for m in range(7):s+=f'    let p{m}=v128_load(products[{m}][$i].as_ptr().add({g*4}).cast());\n'
    for ti,ri,d in [(0,0,'i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)'),(0,1,'i32x4_add(p2,p4)'),(1,0,'i32x4_add(p1,p3)'),(1,1,'i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)')]:
        scale=f'even{g}' if ri==0 else f'odd{g}'
        dest=f'sums[$i][{ti}][{ri}][{g}]'
        s+=f'    {dest}=f32x4_add({dest},f32x4_mul(f32x4_mul(f32x4_convert_i32x4({d}),sx{ti}),{scale}));\n'
    s+='   }\n'
s+='  }}}\n'
for i in range(32):s+=f'  row!({i});\n'
s+=' }\n macro_rules! store {($i:literal)=>{if R>$i {\n'
for ti in range(2):
    s+=f'  {{let token=t+$i*2+{ti};if token<q.rows() {{let dst=out.as_mut_ptr().add(token*rows+r);\n'
    for g in range(C//4):
        s+=f'''   {{let even=sums[$i][{ti}][0][{g}];let odd=sums[$i][{ti}][1][{g}];
   v128_store(dst.add({g*8}).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add({g*8+4}).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}}\n'''
    s+='  } }\n'
s+=' }}}\n'
for i in range(32):s+=f' store!({i});\n'
s+='}\n'
s=s.replace('[[[[zero;4];2];2];R]',f'[[[[zero;{C//4}];2];2];R]')
# Constructor/project bounds establish every full operand/output span. Only
# address arithmetic uses wrapping operations; integer dots remain unchanged.
for old,new in [
 ('|m|a.data.as_ptr().add(m*a.pairs*stride+t/2*stride)', '|m:usize|a.data.as_ptr().add(m.wrapping_mul(a.pairs).wrapping_mul(stride).wrapping_add((t/2).wrapping_mul(stride)))'),
 ('|m|w.data.as_ptr().add(m*(w.rows/2)*stride+r/2*stride)', '|m:usize|w.data.as_ptr().add(m.wrapping_mul(w.rows/2).wrapping_mul(stride).wrapping_add((r/2).wrapping_mul(stride)))'),
 ('t*blocks','t.wrapping_mul(blocks)'),('block*128','block.wrapping_mul(128)'),
 ('($i*2)*blocks+block','($i*2usize).wrapping_mul(blocks).wrapping_add(block)'),
 ('($i*2+1)*blocks+block','($i*2usize+1).wrapping_mul(blocks).wrapping_add(block)'),
 ('token*rows+r','token.wrapping_mul(rows).wrapping_add(r)'),
 ('t+$i*2+','t.wrapping_add($i*2usize+'),
 ]:s=s.replace(old,new)
# Close the wrapping_add in generated token declarations.
s=s.replace('+0;','+0);').replace('+1;','+1);')
s=s.replace('//! Generated tile. Validated constructors bound every pointer span.',
 '//! Generated tile. Validated constructors bound every pointer span.\n//! Wrapping address operations only remove redundant checks; no valid span wraps.')
(ROOT/'scripts/strassen16_bench/src/tile.rs').write_text(s)
