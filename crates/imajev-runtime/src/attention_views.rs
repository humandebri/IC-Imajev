//! Private, shape-checked GQA views. Share KV without rebuilding per-head input.
//! Products may use SIMD; every score retains the scalar left-to-right sum.
use crate::bf;

pub(crate) fn head(q: &[f32], k: &[f32], v: &[f32], n: usize, width: usize, prefix: usize) -> Vec<f32> {
    // The GQA caller has bounded n/width/prefix and checked the complete layout.
    debug_assert_eq!(q.len(), n * width);
    debug_assert_eq!(k.len(), (n + prefix) * width);
    debug_assert_eq!(v.len(), k.len());
    let mut out = vec![0.; n * width];
    let mut scores = Vec::with_capacity(n + prefix);
    let divisor = (width as f32).sqrt();
    for t in 0..n {
        scores.clear();
        let query = &q[t * width..(t + 1) * width];
        for key in k[..(prefix + t + 1) * width].chunks_exact(width) {
            scores.push(bf(dot(query, key) / divisor));
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for s in &mut scores { *s = (*s - maximum).exp(); }
        let denominator = scores.iter().sum::<f32>();
        // Reuse scores instead of allocating another probabilities vector.
        for s in &mut scores { *s = bf(*s / denominator); }
        #[cfg(target_arch = "wasm32")]
        if width % 4 == 0 {
            // Complete shapes above prove each four-lane load/store in the helper.
            unsafe { crate::attention_value_lanes(&scores, v, width, &mut out[t * width..(t + 1) * width]); }
            continue;
        }
        for i in 0..width {
            out[t * width + i] = scores.iter().enumerate().map(|(s, &p)| p * v[s * width + i]).sum();
        }
    }
    for value in &mut out { *value = bf(*value); }
    out
}

fn dot(q: &[f32], k: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    // Identical slices from the checked caller; loads stop before the shared end.
    unsafe { return dot_lanes(q, k); }
    #[cfg(not(target_arch = "wasm32"))]
    q.iter().zip(k).map(|(&a, &b)| a * b).sum()
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_lanes(q: &[f32], k: &[f32]) -> f32 {
    use core::arch::wasm32::*;
    let mut sum = -0.0f32; // std's floating Sum identity, including signed zero.
    let full = q.len() / 4 * 4;
    for i in (0..full).step_by(4) {
        let product = f32x4_mul(v128_load(q.as_ptr().add(i).cast()), v128_load(k.as_ptr().add(i).cast()));
        // Do not use horizontal reduction or FMA: preserve the reference order.
        sum += f32x4_extract_lane::<0>(product);
        sum += f32x4_extract_lane::<1>(product);
        sum += f32x4_extract_lane::<2>(product);
        sum += f32x4_extract_lane::<3>(product);
    }
    for i in full..q.len() { sum += q[i] * k[i]; }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn views_equal_scalar_suffix_for_odd_widths_prefixes_and_real_sizes() {
        for (n, width, prefix) in [(1,1,0),(3,3,1),(7,8,5),(45,256,0),(80,256,45),(87,256,45),(89,256,45),(1,256,131)] {
            let total = n + prefix;
            let q:Vec<_>=(0..n*width).map(|i|bf(((i*17%251)as f32-125.)/128.)).collect();
            let k:Vec<_>=(0..total*width).map(|i|bf(((i*31%239)as f32-119.)/128.)).collect();
            let v:Vec<_>=(0..total*width).map(|i|bf(((i*43%233)as f32-116.)/128.)).collect();
            let r:crate::Request=serde_json::from_value(serde_json::json!({"version":1,"model":"m","pack_hash":"p","input_hash":"i","step":0,"op":"attention_suffix_bf16","tensor":"","dims":[n,width,prefix],"scalars":[],"encoding":"bf16-exact"})).unwrap();
            let expected=crate::execute(&r,&[q.clone(),k.clone(),v.clone()].concat(),&[]).unwrap();
            assert_eq!(head(&q,&k,&v,n,width,prefix).iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
        for zero in [0.0f32,-0.0f32] {let q=vec![zero;256];assert_eq!(dot(&q,&q).to_bits(),q.iter().map(|v|v*v).sum::<f32>().to_bits());}
    }
}
