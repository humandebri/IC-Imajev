//! F32 remainder groups share fixed weights; lanes retain column accumulation order.
use crate::{Result,MAX_FLOATS};
pub(super) fn evaluate(x:&[f32],w:&[f32],n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {
    if n.checked_mul(cols)!=Some(x.len()) || rows.checked_mul(cols)!=Some(w.len())
        || n==0 || rows==0 || cols==0 || n.checked_mul(rows).is_none_or(|s|s>MAX_FLOATS) {
        return Err("matmul shape/output bounds".into());
    }
    if n<4 {return crate::matrix_reference(x,w,n,rows,cols);}
    let main=if n>=128 && rows>=64 {64} else if n>=32 && rows>=64 {32} else {16};
    let mut out=vec![0.;n*rows];let mut packed=vec![0.;cols*main];let mut t=0;
    macro_rules! group {($g:literal,$valid:expr)=>{{
        compute::<$g,false>(x,w,rows,cols,t,$valid,&mut packed,&mut out);
        t+=$valid;
    }}}
    while t+main<=n {
        match main {64=>group!(16,64),32=>group!(8,32),_=>group!(4,16)}
    }
    while t+16<=n {group!(4,16);}
    while t+8<=n {group!(2,8);}
    while t+4<=n {group!(1,4);}
    if t<n {compute::<1,true>(x,w,rows,cols,t,n-t,&mut packed,&mut out);t=n;}
    debug_assert_eq!(t,n);
    Ok(out)
}
fn compute<const G:usize,const PADDED:bool>(x:&[f32],w:&[f32],rows:usize,cols:usize,t:usize,valid:usize,packed:&mut[f32],out:&mut[f32]) {
    let tokens=G*4;
    // Each group rewrites its entire packed prefix. Dummy lanes are initialized
    // to zero and never stored in the observable token-major output.
    let input=&x[t*cols..(t+valid)*cols];
    let prefix=&mut packed[..cols*tokens];
    for (c,column) in prefix.chunks_exact_mut(tokens).enumerate() {
        for (lane,value) in column.iter_mut().enumerate() {
            *value=if PADDED && lane>=valid {0.} else {input[lane*cols+c]};
        }
    }
    let mut r=0;
    while r+16<=rows {
        let dots=dot::<16,G>(packed,&w[r*cols..(r+16)*cols],cols);
        for j in 0..16 {for lane in 0..valid {out[(t+lane)*rows+r+j]=dots[j][lane/4][lane%4];}}
        r+=16;
    }
    while r<rows {
        let dots=dot::<1,G>(packed,&w[r*cols..(r+1)*cols],cols);
        for lane in 0..valid {out[(t+lane)*rows+r]=dots[0][lane/4][lane%4];}
        r+=1;
    }
}
fn dot<const R:usize,const G:usize>(packed:&[f32],weights:&[f32],cols:usize)->[[[f32;4];G];R] {
    #[cfg(target_arch="wasm32")]
    // evaluate checks the complete tensor lengths. compute passes R weight rows
    // and a fully initialized G*4*cols packed prefix to the existing SIMD kernel.
    return unsafe {crate::dot_multi::<R,G>(packed,weights,cols)};
    #[cfg(not(target_arch="wasm32"))]
    {
        let mut acc=[[[0.;4];G];R];
        for c in 0..cols {for r in 0..R {for g in 0..G {for lane in 0..4 {
            acc[r][g][lane]+=packed[c*G*4+g*4+lane]*weights[r*cols+c];
        }}}}
        acc
    }
}
