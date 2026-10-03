#!/usr/bin/env python3
"""Generate an exact one-level Strassen experiment; never deploys itself."""
import pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=ROOT/'crates/imajev-runtime/src/int8_kernel.rs'
s=(ROOT/'artifacts/strassen/baseline.rs').read_text()
a=s.index('    // Keep packed INT8 weights;');b=s.index('    if !out.iter()',a)
s=s[:a]+'''    #[cfg(target_arch="wasm32")]
    let out=unsafe { strassen_project(q,w,scales,rows) };
    #[cfg(not(target_arch="wasm32"))]
    let out={
        let mut out=vec![0.;q.rows*rows];
        for r in (0..rows).step_by(8) {
            for t in (0..q.rows).step_by(8) {project_tile::<8,8>(q,w,scales,rows,t,r,&mut out);}
        }
        out
    };
'''+s[b:]
# Retain the scalar/reference kernel; candidate's integer transformation is Wasm-only.
s+='''
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn strassen_project(q:&QuantizedRows,w:&[i8],sw:&[f32],rows:usize)->Vec<f32> {
    let stride=q.cols/2; let blocks=q.cols/256;let tokens=q.rows.div_ceil(8)*8;
    // Seven operands laid out as [product][pair][block][128].
    let mut a=vec![0i16;7*(tokens/2)*stride];
    let mut b=vec![0i16;7*(rows/2)*stride];
    for pair in 0..tokens/2 {for block in 0..blocks {for k in 0..128 {
        let p=pair*2*q.cols+block*256+k;
        let (a11,a12,a21,a22)=(q.values[p],q.values[p+128],q.values[p+q.cols],q.values[p+q.cols+128]);
        let values=[a11+a22,a21+a22,a11,a22,a11+a12,a21-a11,a12-a22];
        for m in 0..7 {a[m*(tokens/2)*stride+pair*stride+block*128+k]=values[m];}
    }}}
    for pair in 0..rows/2 {for block in 0..blocks {for k in 0..128 {
        let p=pair*2*q.cols+block*256+k;
        let (b11,b21,b12,b22)=(w[p] as i16,w[p+128] as i16,w[p+q.cols] as i16,w[p+q.cols+128] as i16);
        let values=[b11+b22,b11,b12-b22,b21-b11,b22,b11+b12,b21+b22];
        for m in 0..7 {b[m*(rows/2)*stride+pair*stride+block*128+k]=values[m];}
    }}}
    let mut out=vec![0.;q.rows*rows];
    for r in (0..rows).step_by(8) {
        let mut t=0;
        while t<q.rows {
            if t+64<=tokens {strassen_tile::<32>(&a,&b,q,sw,rows,t,r,&mut out);t+=64;}
            else if t+32<=tokens {strassen_tile::<16>(&a,&b,q,sw,rows,t,r,&mut out);t+=32;}
            else if t+16<=tokens {strassen_tile::<8>(&a,&b,q,sw,rows,t,r,&mut out);t+=16;}
            else {strassen_tile::<4>(&a,&b,q,sw,rows,t,r,&mut out);t+=8;}
        }
    }
    out
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn strassen_tile<const R:usize>(a:&[i16],b:&[i16],q:&QuantizedRows,sw:&[f32],rows:usize,t:usize,r:usize,out:&mut[f32]) {
    let stride=q.cols/2;let tokens=q.rows.div_ceil(8)*8;let blocks=q.cols/256;
    let mut sums=[[[0f32;8];2];R];
    for block in 0..blocks {
        let products:[[[i32;4];R];7]=core::array::from_fn(|m|{
            strassen_dot::<R>(a.as_ptr().add(m*(tokens/2)*stride+t/2*stride+block*128),b.as_ptr().add(m*(rows/2)*stride+r/2*stride+block*128),stride)
        });
        for i in 0..R {for j in 0..4 {
            let p=|m:usize|products[m][i][j];
            let d=[[p(0)+p(3)-p(4)+p(6),p(2)+p(4)],[p(1)+p(3),p(0)-p(1)+p(2)+p(5)]];
            for ti in 0..2 {for ri in 0..2 {
                sums[i][ti][j*2+ri]+=(d[ti][ri] as f32*q.scales[(t+i*2+ti)*blocks+block])*sw[r+j*2+ri];
            }}
        }}
    }
    for i in 0..R {for ti in 0..2 {if t+i*2+ti<q.rows {for j in 0..8 {out[(t+i*2+ti)*rows+r+j]=sums[i][ti][j];}}}}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn strassen_dot<const R:usize>(a:*const i16,b:*const i16,stride:usize)->[[i32;4];R] {
    use core::arch::wasm32::*;
    let weights:[[v128;16];4]=core::array::from_fn(|j|core::array::from_fn(|k|v128_load(b.add(j*stride+k*8).cast())));
    let mut out=[[0i32;4];R];
'''
# Explicit row/column expression lets LLVM share loads as in the baseline.
def tree(xs):
 if len(xs)==1:return xs[0]
 n=len(xs)//2;return f'i32x4_add({tree(xs[:n])},{tree(xs[n:])})'
s+='    macro_rules! row {($i:literal)=>{if R>$i {\n'
s+='        let input:[v128;16]=core::array::from_fn(|k|v128_load(a.add($i*stride+k*8).cast()));\n'
for j in range(4):
 expr=tree([f'i32x4_dot_i16x8(input[{k}],weights[{j}][{k}])' for k in range(16)])
 s+=f'        {{let v={expr};let halves=i32x4_add(v,i32x4_shuffle::<2,3,0,1>(v,v));let total=i32x4_add(halves,i32x4_shuffle::<1,0,3,2>(halves,halves));out[$i][{j}]=i32x4_extract_lane::<0>(total);}}\n'
s+='    }}};\n'+''.join(f'    row!({i});\n' for i in range(32))+'    out\n}\n'
p.write_text(s)
