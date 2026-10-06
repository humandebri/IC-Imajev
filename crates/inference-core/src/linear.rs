//! Column-order F32 projections with independent token SIMD lanes.
use crate::{Result, MAX_FLOATS};

pub fn matrix_reference(
    x: &[f32],
    w: &[f32],
    n: usize,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if n.checked_mul(cols) != Some(x.len()) || rows.checked_mul(cols) != Some(w.len()) {
        return Err("matmul shape".into());
    }
    if n == 0 || rows == 0 || cols == 0 || n.checked_mul(rows).is_none_or(|v| v > MAX_FLOATS) {
        return Err("matmul output bounds".into());
    }
    let mut out = vec![0.0; n * rows];
    for t in 0..n {
        for r in 0..rows {
            let mut v = 0.0;
            for c in 0..cols {
                v += x[t * cols + c] * w[r * cols + c];
            }
            out[t * rows + r] = v;
        }
    }
    Ok(out)
}
/// Original kernel retained for same-Wasm diagnostic comparisons.
pub fn matrix_baseline(
    x: &[f32],
    w: &[f32],
    n: usize,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if n.checked_mul(cols) != Some(x.len())
        || rows.checked_mul(cols) != Some(w.len())
        || n == 0
        || rows == 0
        || cols == 0
        || n.checked_mul(rows).is_none_or(|v| v > MAX_FLOATS)
    {
        return Err("matmul shape/output bounds".into());
    }
    if n < 4 {
        return matrix_reference(x, w, n, rows, cols);
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Shape-calibrated tile sizes; larger tiles lose on short/small matrices.
        return Ok(unsafe {
            if n >= 128 && rows >= 64 {
                matrix_multi::<16>(x, w, n, rows, cols)
            } else if n >= 32 && rows >= 64 {
                matrix_multi::<8>(x, w, n, rows, cols)
            } else {
                matrix_multi::<4>(x, w, n, rows, cols)
            }
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut out = vec![0.0; n * rows];
        let mut packed = vec![0.0; cols * 4];
        for t in (0..n / 4 * 4).step_by(4) {
            for c in 0..cols {
                for lane in 0..4 {
                    packed[c * 4 + lane] = x[(t + lane) * cols + c];
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                for r in 0..rows {
                    let weights = &w[r * cols..(r + 1) * cols];
                    #[cfg(target_arch = "wasm32")]
                    let sums = unsafe { dot_four(&packed, weights) };
                    #[cfg(not(target_arch = "wasm32"))]
                    let sums = {
                        let mut sums = [0.0; 4];
                        for c in 0..cols {
                            for lane in 0..4 {
                                sums[lane] += packed[c * 4 + lane] * weights[c];
                            }
                        }
                        sums
                    };
                    for lane in 0..4 {
                        out[(t + lane) * rows + r] = sums[lane];
                    }
                }
            }
        }
        for t in n / 4 * 4..n {
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += x[t * cols + c] * w[r * cols + c];
                }
                out[t * rows + r] = sum;
            }
        }
        Ok(out)
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_four(packed: &[f32], weights: &[f32]) -> [f32; 4] {
    use core::arch::wasm32::*;
    // matrix validates lengths; each load spans exactly four adjacent F32 lanes.
    let mut sum = f32x4_splat(0.0);
    for (c, &weight) in weights.iter().enumerate() {
        let input = unsafe { v128_load(packed.as_ptr().add(c * 4).cast()) };
        sum = f32x4_add(sum, f32x4_mul(input, f32x4_splat(weight)));
    }
    let mut result = [0.0; 4];
    unsafe {
        v128_store(result.as_mut_ptr().cast(), sum);
    }
    result
}
/// Independent output rows share input loads; accumulation order stays unchanged.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_four_rows<const R: usize>(
    packed: &[f32],
    weights: &[f32],
    cols: usize,
) -> [[f32; 4]; R] {
    use core::arch::wasm32::*;
    let mut acc = [f32x4_splat(0.); R];
    let row_ptrs: [*const f32; R] = core::array::from_fn(|r| weights.as_ptr().add(r * cols));
    let mut input_ptr = packed.as_ptr();
    for c in 0..cols {
        let input = v128_load(input_ptr.cast());
        input_ptr = input_ptr.add(4);
        macro_rules! row {($($r:literal),*)=>{$(if R>$r {
            acc[$r]=f32x4_add(acc[$r],f32x4_mul(input,f32x4_splat(*row_ptrs[$r].add(c))));
        })*};}
        row!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
    let mut out = [[0.; 4]; R];
    for r in 0..R {
        v128_store(out[r].as_mut_ptr().cast(), acc[r]);
    }
    out
}
/// Share each weight broadcast across independent token vectors. Each lane
/// still visits columns in scalar order, with separate F32 multiply and add.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn matrix_multi<const G: usize>(
    x: &[f32],
    w: &[f32],
    n: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let tokens = G * 4;
    let mut out = vec![0.0; n * rows];
    let mut packed = vec![0.0; cols * tokens];
    let mut t = 0;
    while t + tokens <= n {
        for c in 0..cols {
            for lane in 0..tokens {
                packed[c * tokens + lane] = x[(t + lane) * cols + c];
            }
        }
        let mut r = 0;
        while r + 16 <= rows {
            let sums = dot_multi::<16, G>(&packed, &w[r * cols..(r + 16) * cols], cols);
            for j in 0..16 {
                for lane in 0..tokens {
                    out[(t + lane) * rows + r + j] = sums[j][lane / 4][lane % 4];
                }
            }
            r += 16;
        }
        while r < rows {
            let sums = dot_multi::<1, G>(&packed, &w[r * cols..(r + 1) * cols], cols);
            for lane in 0..tokens {
                out[(t + lane) * rows + r] = sums[0][lane / 4][lane % 4];
            }
            r += 1;
        }
        t += tokens;
    }
    // Keep the four-lane fast path for the final complete vector.
    while t + 4 <= n {
        for c in 0..cols {
            for lane in 0..4 {
                packed[c * 4 + lane] = x[(t + lane) * cols + c];
            }
        }
        let mut r = 0;
        while r + 16 <= rows {
            let sums = dot_four_rows::<16>(&packed, &w[r * cols..(r + 16) * cols], cols);
            for j in 0..16 {
                for lane in 0..4 {
                    out[(t + lane) * rows + r + j] = sums[j][lane];
                }
            }
            r += 16;
        }
        while r < rows {
            let sums = dot_four(&packed, &w[r * cols..(r + 1) * cols]);
            for lane in 0..4 {
                out[(t + lane) * rows + r] = sums[lane];
            }
            r += 1;
        }
        t += 4;
    }
    while t < n {
        for r in 0..rows {
            let mut sum = 0.0;
            for c in 0..cols {
                sum += x[t * cols + c] * w[r * cols + c];
            }
            out[t * rows + r] = sum;
        }
        t += 1;
    }
    out
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
/// Raw SIMD entry used by model-specific packing implementations.
///
/// # Safety
/// `R` must be in 1..=16 and `G` must be positive. `packed` must contain
/// at least `cols * G * 4` initialized elements and `weights` at least
/// `cols * R` elements, with products fitting usize. SIMD128 must be available.
#[doc(hidden)]
pub unsafe fn dot_multi<const R: usize, const G: usize>(
    packed: &[f32],
    weights: &[f32],
    cols: usize,
) -> [[[f32; 4]; G]; R] {
    use core::arch::wasm32::*;
    let mut acc = [[f32x4_splat(0.); G]; R];
    let row_ptrs: [*const f32; R] = core::array::from_fn(|r| weights.as_ptr().add(r * cols));
    let mut input_ptr = packed.as_ptr();
    for c in 0..cols {
        let inputs: [v128; G] = core::array::from_fn(|g| v128_load(input_ptr.add(g * 4).cast()));
        input_ptr = input_ptr.add(G * 4);
        macro_rules! row {($($r:literal),*)=>{$(if R>$r {
            let weight = f32x4_splat(*row_ptrs[$r].add(c));
            for g in 0..G { acc[$r][g] = f32x4_add(acc[$r][g],f32x4_mul(inputs[g],weight)); }
        })*};}
        row!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
    let mut out = [[[0.; 4]; G]; R];
    for r in 0..R {
        for g in 0..G {
            v128_store(out[r][g].as_mut_ptr().cast(), acc[r][g]);
        }
    }
    out
}
