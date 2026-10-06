#!/usr/bin/env python3
"""Generate output128 LUT: unpack fixed weight indices once for all tokens.

Weight cache stays W4. Expanded indices live in Wasm function locals only;
input LUT loads are shared by eight output16 tiles. Exact I32/F32 order retained.
"""
import argparse, pathlib, shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);ap.add_argument('--full-query',action='store_true');a=ap.parse_args();d=ROOT/a.directory
    d.mkdir(parents=True,exist_ok=True)
    if any(d.iterdir()):raise ValueError('Fresh directory required')
    source=ROOT/'scripts/w4_lut_bench/src';text=(source/'kernel.rs').read_text()
    # Preserve the checked public entry point; generated body requires output128.
    text=text.replace('assert!(rows%16==0);','assert!(rows%128==0);')
    start=text.index('unsafe fn lut_simd(');end=text.index('#[cfg(test)]',start)
    lines=['unsafe fn lut_simd(sx:&[f32],packed:&[u8],sw:&[f32],n:usize,rows:usize,cols:usize,tab:&[u8])->Vec<f32>{',
    'use core::arch::wasm32::*;',
    '#[inline(always)] unsafe fn read(lo:v128,hi:v128,index:v128)->(v128,v128){let l=i8x16_swizzle(lo,index);let h=i8x16_swizzle(hi,index);',
    '(i8x16_shuffle::<0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23>(l,h),i8x16_shuffle::<8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31>(l,h))}',
    'let mut out=vec![0.;n*rows];let blocks=cols/256;let mask=i8x16_splat(15);',
    'for r in (0..rows).step_by(128){for b in 0..blocks{']
    for tile in range(8):
        lines.append(f'let weights{tile}=packed.as_ptr().add(((r/16+{tile})*(cols/4)+b*64)*32);')
        for g in range(64):
            name=f'w{tile}_{g}';lines += [f'let p=v128_load(weights{tile}.add({g*32}).cast());let z=v128_load(weights{tile}.add({g*32+16}).cast());',
            f'let {name}_0=v128_and(p,mask);let {name}_1=u8x16_shr(p,4);let {name}_2=v128_and(z,mask);let {name}_3=u8x16_shr(z,4);']
    lines.append('for t in 0..n{let table=tab.as_ptr().add((t*cols/4+b*64)*32);')
    for tile in range(8):lines.append(f'let mut a{tile}=i32x4_splat(0);let mut c{tile}=a{tile};let mut d{tile}=a{tile};let mut e{tile}=a{tile};')
    for g in range(64):
        lines.append(f'let lo=v128_load(table.add({g*32}).cast());let hi=v128_load(table.add({g*32+16}).cast());')
        for tile in range(8):
            name=f'w{tile}_{g}';lines += ['{',f'let (v0,v1)=read(lo,hi,{name}_0);let (w0,w1)=read(lo,hi,{name}_1);let (x0,x1)=read(lo,hi,{name}_2);let (y0,y1)=read(lo,hi,{name}_3);',
            'let l=i16x8_sub(i16x8_add(i16x8_add(v0,i16x8_shl(w0,1)),i16x8_shl(x0,2)),i16x8_shl(y0,3));',
            'let h=i16x8_sub(i16x8_add(i16x8_add(v1,i16x8_shl(w1,1)),i16x8_shl(x1,2)),i16x8_shl(y1,3));',
            f'a{tile}=i32x4_add(a{tile},i32x4_extend_low_i16x8(l));c{tile}=i32x4_add(c{tile},i32x4_extend_high_i16x8(l));d{tile}=i32x4_add(d{tile},i32x4_extend_low_i16x8(h));e{tile}=i32x4_add(e{tile},i32x4_extend_high_i16x8(h));','}']
    for tile in range(8):
        lines += ['{let mut dots=[0i32;16];',f'v128_store(dots.as_mut_ptr().cast(),a{tile});v128_store(dots.as_mut_ptr().add(4).cast(),c{tile});v128_store(dots.as_mut_ptr().add(8).cast(),d{tile});v128_store(dots.as_mut_ptr().add(12).cast(),e{tile});',
        f'for lane in 0..16{{out[t*rows+r+{tile*16}+lane]+=(dots[lane]as f32*sx[t*blocks+b])*sw[(r+{tile*16}+lane)*blocks+b];}}','}']
    lines.append('}}}out}\n')
    (d/'kernel.rs').write_text(text[:start]+'\n'.join(lines)+text[end:])
    lib=(source/'lib.rs').read_text()
    if a.full_query:
        for old,new in [('f.sealed&&method<16','f.sealed&&method<=16'),('let start=(method as usize/2)*1024;','let start=if method==16{0}else{(method as usize/2)*1024};'),('let rows=1024.min(f.rows-start);let use_lut=method%2==1;','let rows=if method==16{f.rows}else{1024.min(f.rows-start)};let use_lut=method==16||method%2==1;')]:
            assert lib.count(old)==1;lib=lib.replace(old,new)
    (d/'lib.rs').write_text(lib)
    print(d)
if __name__=='__main__':main()
