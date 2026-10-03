//! Fixed K4/output-pair layout, retaining exact block256 integer dots/scales.
//! Weight bytes remain INT8 and unchanged in count; input duplication is temporary.
use imajev_runtime::{int8_kernel::QuantizedRows, Result};
pub struct PairWeights { data:Vec<i8>, rows:usize, cols:usize }
pub struct PairInput { data:Vec<i16>, rows:usize, cols:usize }
pub fn prepare_weights(w:&[i8],rows:usize,cols:usize)->Result<PairWeights> {
    if rows==0 || rows%32!=0 || cols==0 || cols%256!=0 || rows.checked_mul(cols)!=Some(w.len()) || w.len()>30_000_000 {return Err("pair weight shape".into());}
    let mut data=Vec::with_capacity(w.len());
    for r in (0..rows).step_by(2) {for c in (0..cols).step_by(4) {
        data.extend_from_slice(&w[r*cols+c..r*cols+c+4]);
        data.extend_from_slice(&w[(r+1)*cols+c..(r+1)*cols+c+4]);
    }}
    Ok(PairWeights{data,rows,cols})
}
pub fn prepare_input(q:&QuantizedRows)->PairInput {
    let mut data:Vec<i16>=Vec::with_capacity(q.values().len()*2);
    #[cfg(target_arch="wasm32")]
    // QuantizedRows has complete block256 rows and initialized token padding.
    // Each source load reads four I16 values; one SIMD store writes both copies
    // into the allocated spare capacity. All new elements are initialized once.
    unsafe {
        duplicate_simd(q.values(),data.as_mut_ptr());
        data.set_len(q.values().len()*2);
    }
    #[cfg(not(target_arch="wasm32"))]
    for chunk in q.values().chunks_exact(4) {data.extend_from_slice(chunk);data.extend_from_slice(chunk);}
    PairInput{data,rows:q.rows(),cols:q.cols()}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn duplicate_simd(input:&[i16],output:*mut i16) {
    use core::arch::wasm32::*;
    for i in (0..input.len()).step_by(4) {
        v128_store(output.add(i*2).cast(),v128_load64_splat(input.as_ptr().add(i).cast()));
    }
}
pub fn project(q:&QuantizedRows,p:&PairInput,w:&PairWeights,scales:&[f32])->Result<Vec<f32>> {
    if !(81..=88).contains(&q.rows()) || q.rows()!=p.rows || q.cols()!=p.cols || p.cols!=w.cols || scales.len()!=w.rows || !scales.iter().all(|v|v.is_finite()&&*v>0.) || q.rows()*w.rows>900_000 {return Err("pair projection shape/scales".into());}
    let mut out=vec![0.;q.rows()*w.rows];
    for r in (0..w.rows).step_by(32) {for t in [0,44] {
        let mut sums=[[0f32;32];44];
        for block in 0..w.cols/256 {
            #[cfg(target_arch="wasm32")]
            // Opaque constructors establish packed strides, complete padded88
            // rows, original scales and exact eight-byte weight load bounds.
            unsafe {crate::kernel::accumulate(p.data.as_ptr().add(t*w.cols*2),w.data.as_ptr().add(r*w.cols),w.cols,block*256,q.scales().as_ptr().add(t*(w.cols/256)+block),w.cols/256,scales.as_ptr().add(r),&mut sums);}
            #[cfg(not(target_arch="wasm32"))]
            for i in 0..44 {for j in 0..32 {
                let mut dot=0i32;
                for c in block*256..(block+1)*256 {
                    let wi=((r+j)/2)*w.cols*2+(c/4)*8+((r+j)%2)*4+c%4;
                    let qi=(t+i)*w.cols*2+(c/4)*8+c%4;
                    dot+=p.data[qi]as i32*w.data[wi]as i32;
                }
                sums[i][j]+=(dot as f32*q.scales()[(t+i)*(w.cols/256)+block])*scales[r+j];
            }}
        }
        for i in 0..44 {if t+i<q.rows() {out[(t+i)*w.rows+r..(t+i)*w.rows+r+32].copy_from_slice(&sums[i]);}}
    }}
    if out.iter().any(|v|!v.is_finite()){return Err("pair projection output".into());}Ok(out)
}
#[cfg(test)]mod tests {use super::*;
    #[test]fn signed_extremes_and_block_scales(){for n in [81,87,88] {for cols in [256,512] {let rows=32;
        let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,-0.,0.,1.,-1.,0.5,-0.5][i%8]*(1+i/256%3)as f32).collect();
        let w:Vec<i8>=(0..rows*cols).map(|i|[-128,127,-127,0,1,-1][i%6]).collect();let scales:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();
        let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let pw=prepare_weights(&w,rows,cols).unwrap();let pq=prepare_input(&q);assert_eq!(pw.data.len(),w.len());
        let old=imajev_runtime::int8_kernel::project_column32_balanced(&q,&w,&scales,rows).unwrap();let new=project(&q,&pq,&pw,&scales).unwrap();assert!(old.iter().zip(new).all(|(a,b)|a.to_bits()==b.to_bits()));
    }}}
}
