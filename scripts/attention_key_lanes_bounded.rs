//! Four independent key lanes, with each dot accumulated in original K order.
pub fn scores(q:&[f32],k:&[f32],n:usize,width:usize,prefix:usize)->Vec<f32>{
    assert!(n>0 && n<=132 && prefix<=132-n && width>0 && width<=256 && q.len()==n*width && k.len()==(n+prefix)*width);
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
        *transposed.get_unchecked_mut((key/4*width+j)*4+key%4)=*k.get_unchecked(key*width+j);
    }}
    let divisor=(width as f32).sqrt();
    let mut out=Vec::with_capacity(n*prefix+n*(n+1)/2);
    for t in 0..n {
        let used=prefix+t+1;
        for block in 0..used.div_ceil(4) {
            let mut sums=f32x4_splat(-0.0f32);
            let qp=q.as_ptr().add(t*width);
            let kp=transposed.as_ptr().add(block*width*4);
            let mut j=0;
            while j+8<=width {
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+0)),v128_load(kp.add((j+0)*4).cast())));
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+1)),v128_load(kp.add((j+1)*4).cast())));
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+2)),v128_load(kp.add((j+2)*4).cast())));
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+3)),v128_load(kp.add((j+3)*4).cast())));
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+4)),v128_load(kp.add((j+4)*4).cast())));
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+5)),v128_load(kp.add((j+5)*4).cast())));
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+6)),v128_load(kp.add((j+6)*4).cast())));
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j+7)),v128_load(kp.add((j+7)*4).cast())));
                j+=8;
            }
            while j<width {
                sums=f32x4_add(sums,f32x4_mul(f32x4_splat(*qp.add(j)),v128_load(kp.add(j*4).cast())));
                j+=1;
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
