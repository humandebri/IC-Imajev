//! Experimental per-token A8 scales and full-column I32 accumulation.
//! This changes arithmetic from block256; it is not enabled by the default graph.
use crate::Result;
pub struct TokenRows {
    pub values: Vec<i16>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}
pub fn quantize(x: &[f32], rows: usize, cols: usize) -> Result<TokenRows> {
    if rows == 0
        || rows > 512
        || cols == 0
        || cols > 131072
        || cols % 256 != 0
        || rows.checked_mul(cols) != Some(x.len())
        || !x.iter().all(|v| v.is_finite())
    {
        return Err("token quantization bounds".into());
    }
    let mut values = vec![0i16; rows.div_ceil(8) * 8 * cols];
    let mut scales = vec![1.; rows.div_ceil(8) * 8];
    for r in 0..rows {
        #[cfg(target_arch = "wasm32")]
        {
            scales[r] = unsafe {
                quantize_simd(
                    x.as_ptr().add(r * cols),
                    values.as_mut_ptr().add(r * cols),
                    cols,
                )
            };
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let peak = x[r * cols..(r + 1) * cols]
                .iter()
                .map(|v| v.abs())
                .fold(0., f32::max);
            let s = if peak == 0. {
                1.
            } else {
                (peak / 127.).max(f32::from_bits(1))
            };
            scales[r] = s;
            for c in 0..cols {
                values[r * cols + c] =
                    (x[r * cols + c] / s).round_ties_even().clamp(-127., 127.) as i16;
            }
        }
    }
    Ok(TokenRows {
        values,
        scales,
        rows,
        cols,
    })
}
pub fn project(q: &TokenRows, w: &[i8], sw: &[f32], rows: usize) -> Result<Vec<f32>> {
    let cols = q.cols;
    let padded = q.rows.div_ceil(8) * 8;
    if q.rows == 0
        || q.rows > 512
        || cols == 0
        || cols > 131072
        || cols % 256 != 0
        || q.values.len() != padded * cols
        || q.scales.len() != padded
        || !q.values.iter().all(|v| (-127..=127).contains(v))
        || !q.scales.iter().all(|v| v.is_finite() && *v > 0.)
        || rows == 0
        || rows % 8 != 0
        || rows.checked_mul(cols) != Some(w.len())
        || w.len() > 30_000_000
        || sw.len() != rows
        || !sw.iter().all(|v| v.is_finite() && *v > 0.)
        || q.rows
            .checked_mul(rows)
            .is_none_or(|v| v > crate::MAX_FLOATS)
    {
        return Err("token integer projection bounds".into());
    }
    // cols <= 131072 bounds the entire signed sum even for weight -128.
    let mut out = vec![0.; q.rows * rows];
    for r in (0..rows).step_by(8) {
        let mut t = 0;
        while t < q.rows {
            macro_rules! tile {
                ($n:literal) => {{
                    project_tile::<$n, 8>(q, w, sw, rows, t, r, &mut out);
                    t += $n;
                }};
            }
            let remaining = if q.rows < 8 { q.rows - t } else { padded - t };
            if remaining >= 64 {
                tile!(64);
            } else if remaining >= 32 {
                tile!(32);
            } else if remaining >= 16 {
                tile!(16);
            } else if remaining >= 8 {
                tile!(8);
            } else if remaining >= 4 {
                tile!(4);
            } else if remaining >= 2 {
                tile!(2);
            } else {
                tile!(1);
            }
        }
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err("token projection nonfinite".into());
    }
    Ok(out)
}
fn project_tile<const R: usize, const C: usize>(
    q: &TokenRows,
    w: &[i8],
    sw: &[f32],
    rows: usize,
    t: usize,
    r: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "wasm32")]
    let dots = unsafe {
        dot_tile::<R, C>(
            q.values.as_ptr().add(t * q.cols),
            w.as_ptr().add(r * q.cols),
            q.cols,
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let dots = {
        let mut d = [[0i32; C]; R];
        for i in 0..R {
            for j in 0..C {
                for c in 0..q.cols {
                    d[i][j] +=
                        q.values[(t + i) * q.cols + c] as i32 * w[(r + j) * q.cols + c] as i32;
                }
            }
        }
        d
    };
    for i in 0..R {
        if t + i < q.rows {
            for j in 0..C {
                out[(t + i) * rows + r + j] = (dots[i][j] as f32 * q.scales[t + i]) * sw[r + j];
            }
        }
    }
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn quantize_simd(x: *const f32, q: *mut i16, cols: usize) -> f32 {
    use core::arch::wasm32::*;
    let mut peak = f32x4_splat(0.);
    let mask = i32x4_splat(0x7fffffff);
    for c in (0..cols).step_by(4) {
        peak = f32x4_max(peak, v128_and(v128_load(x.add(c).cast()), mask));
    }
    peak = f32x4_max(peak, i32x4_shuffle::<2, 3, 0, 1>(peak, peak));
    peak = f32x4_max(peak, i32x4_shuffle::<1, 0, 3, 2>(peak, peak));
    let max = f32x4_extract_lane::<0>(peak);
    let scale = if max == 0. {
        1.
    } else {
        (max / 127.).max(f32::from_bits(1))
    };
    let s = f32x4_splat(scale);
    let low = f32x4_splat(-127.);
    let high = f32x4_splat(127.);
    for c in (0..cols).step_by(8) {
        let a = f32x4_min(
            high,
            f32x4_max(low, f32x4_nearest(f32x4_div(v128_load(x.add(c).cast()), s))),
        );
        let b = f32x4_min(
            high,
            f32x4_max(
                low,
                f32x4_nearest(f32x4_div(v128_load(x.add(c + 4).cast()), s)),
            ),
        );
        v128_store(
            q.add(c).cast(),
            i16x8_narrow_i32x4(i32x4_trunc_sat_f32x4(a), i32x4_trunc_sat_f32x4(b)),
        );
    }
    scale
}
// Whole-row integer accumulation has no intermediate F32 scaling.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_tile<const R: usize, const C: usize>(
    q: *const i16, w: *const i8, cols: usize,
) -> [[i32; C]; R] {
    let mut result = [[0i32; C]; R];
    for start in (0..cols).step_by(256) {
        let part = crate::int8_tile::block::<R, C>(q, w, cols, start);
        for row in 0..R { for column in 0..C {
            result[row][column] = result[row][column].wrapping_add(part[row][column]);
        } }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn short_rows_match_whole_column_scalar_sum() {
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 9, 63, 64, 65] {
            let x: Vec<f32> = (0..rows * 512)
                .map(|i| ((i * 37 % 211) as f32 - 105.) / 31.)
                .collect();
            let q = quantize(&x, rows, 512).unwrap();
            let w: Vec<i8> = (0..8 * 512).map(|i| (i % 256) as u8 as i8).collect();
            let sw = vec![0.013; 8];
            let y = project(&q, &w, &sw, 8).unwrap();
            for t in 0..rows {
                for r in 0..8 {
                    let dot: i32 = (0..512)
                        .map(|c| q.values[t * 512 + c] as i32 * w[r * 512 + c] as i32)
                        .sum();
                    assert_eq!(
                        y[t * 8 + r].to_bits(),
                        ((dot as f32 * q.scales[t]) * sw[r]).to_bits()
                    );
                }
            }
        }
    }
    #[test]
    fn ties_zero_and_signed_i32_limit_are_bounded() {
        let mut x = vec![0.; 256];
        x[0] = 127.;
        x[1] = -2.5;
        x[2] = 2.5;
        let q = quantize(&x, 1, 256).unwrap();
        assert_eq!(&q.values[..3], &[127, -2, 2]);
        let z = quantize(&vec![0.; 256], 1, 256).unwrap();
        assert_eq!(z.scales[0], 1.);
        assert!(quantize(&vec![0.; 131328], 1, 131328).is_err());
        let cols = 131072;
        let mut q = quantize(&vec![127.; cols], 1, cols).unwrap();
        let y = project(&q, &vec![-128; 8 * cols], &[1.; 8], 8).unwrap();
        assert!(y.iter().all(|v| *v == (-128i64 * 127 * cols as i64) as f32));
        q.values.pop();
        assert!(project(&q, &vec![-128; 8 * cols], &[1.; 8], 8).is_err());
    }
}
