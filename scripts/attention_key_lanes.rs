//! Four independent key lanes, with each dot accumulated in original K order.
pub fn scores(q:&[f32],k:&[f32],n:usize,width:usize,prefix:usize)->Vec<f32>{
    assert!(width>0 && q.len()==n*width && k.len()==(n+prefix)*width);
    #[cfg(target_arch="wasm32")]
    unsafe { return scores_simd(q,k,n,width,prefix); }
    #[cfg(not(target_arch="wasm32"))]
    {
        let mut out=Vec::new();let divisor=(width as f32).sqrt();
        for t in 0..n {for key in 0..prefix+t+1 {
            let mut sum=-0.0f32;
            for j in 0..width {sum+=q[t*width+j]*k[key*width+j];}
            out.push(crate::bf(sum/divisor));
        }}
        out
    }
}

#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn scores_simd(q:&[f32],k:&[f32],n:usize,width:usize,prefix:usize)->Vec<f32>{
    use core::arch::wasm32::*;
    let total=n+prefix;
    let blocks=total.div_ceil(4);
    // Final block padding is read only by lanes whose results are discarded.
    let mut transposed=vec![0f32;blocks*width*4];
    for key in 0..total {for j in 0..width {
        transposed[(key/4*width+j)*4+key%4]=k[key*width+j];
    }}
    let divisor=(width as f32).sqrt();
    let mut out=Vec::with_capacity(n*prefix+n*(n+1)/2);
    for t in 0..n {
        let used=prefix+t+1;
        for block in 0..used.div_ceil(4) {
            let mut sums=f32x4_splat(-0.0f32);
            for j in 0..width {
                let a=f32x4_splat(q[t*width+j]);
                let b=v128_load(transposed.as_ptr().add((block*width+j)*4).cast());
                sums=f32x4_add(sums,f32x4_mul(a,b));
            }
            let lanes=[f32x4_extract_lane::<0>(sums),f32x4_extract_lane::<1>(sums),
                       f32x4_extract_lane::<2>(sums),f32x4_extract_lane::<3>(sums)];
            for lane in 0..(used-block*4).min(4) {
                out.push(crate::bf(lanes[lane]/divisor));
            }
        }
    }
    out
}
