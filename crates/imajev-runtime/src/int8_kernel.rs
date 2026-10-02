//! Experimental W8/A8 projection: block256 activation scales, per-row weight scales.
//! Integer dots accumulate exactly in I32; scaling/summing blocks uses F32.
use crate::Result;
#[derive(Debug)]
pub struct QuantizedRows {
    pub values: Vec<i16>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}
pub fn quantize_rows(x: &[f32], rows: usize, cols: usize) -> Result<QuantizedRows> {
    if rows == 0
        || rows > 512
        || cols > 262144
        || cols == 0
        || cols % 256 != 0
        || rows.checked_mul(cols) != Some(x.len())
        || !x.iter().all(|v| v.is_finite())
    {
        return Err("integer projection input".into());
    }
    let padded = rows.div_ceil(8) * 8;
    let mut q = vec![0i16; padded * cols];
    let mut scales = vec![1f32; padded * (cols / 256)];
    for r in 0..rows {
        for block in 0..cols / 256 {
            let start = r * cols + block * 256;
            #[cfg(target_arch = "wasm32")]
            {
                scales[r * (cols / 256) + block] = unsafe {
                    crate::quantize_simd::block(x.as_ptr().add(start), q.as_mut_ptr().add(start))
                };
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let v = &x[start..start + 256];
                let max = v.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let scale = if max == 0. {
                    1.
                } else {
                    (max / 127.).max(f32::from_bits(1))
                };
                scales[r * (cols / 256) + block] = scale;
                for i in 0..256 {
                    q[start + i] = (v[i] / scale).round_ties_even().clamp(-127., 127.) as i16;
                }
            }
        }
    }
    Ok(QuantizedRows {
        values: q,
        scales,
        rows,
        cols,
    })
}
pub fn project(q: &QuantizedRows, w: &[i8], scales: &[f32], rows: usize) -> Result<Vec<f32>> {
    let cols = q.cols;
    if q.rows == 0
        || q.rows > 512
        || cols == 0
        || cols > 262144
        || cols % 256 != 0
        || q.values.len() != q.rows.div_ceil(8) * 8 * cols
        || q.scales.len() != q.rows.div_ceil(8) * 8 * (cols / 256)
        || !q.values.iter().all(|v| (-127..=127).contains(v))
        || !q.scales.iter().all(|v| v.is_finite() && *v > 0.)
    {
        return Err("integer projection activations".into());
    }
    if rows == 0
        || rows % 8 != 0
        || rows.checked_mul(cols) != Some(w.len())
        || w.len() > 30_000_000
        || q.rows
            .checked_mul(rows)
            .is_none_or(|v| v > crate::MAX_FLOATS)
        || scales.len() != rows
        || !scales.iter().all(|v| v.is_finite() && *v > 0.)
    {
        return Err("integer projection weights".into());
    }
    // Keep packed INT8 weights; extend each shared 8-byte load in registers.
    let weights = w;
    let mut out = vec![0.; q.rows * rows];
    let mut r = 0;
    while r < rows {
        let mut t = 0;
        while t < q.rows {
            let padded = q.rows.div_ceil(8) * 8;
            macro_rules! dispatch {
                ($c:literal) => {
                    if t + 64 <= padded {
                        project_tile::<64, $c>(q, &weights, scales, rows, t, r, &mut out);
                        t += 64;
                    } else if t + 32 <= padded {
                        project_tile::<32, $c>(q, &weights, scales, rows, t, r, &mut out);
                        t += 32;
                    } else if t + 16 <= padded {
                        project_tile::<16, $c>(q, &weights, scales, rows, t, r, &mut out);
                        t += 16;
                    } else {
                        project_tile::<8, $c>(q, &weights, scales, rows, t, r, &mut out);
                        t += 8;
                    }
                };
            }
            dispatch!(8);
        }
        r += 8;
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err("integer projection output".into());
    }
    Ok(out)
}
fn project_tile<const R: usize, const C: usize>(
    q: &QuantizedRows,
    weights: &[i8],
    scales: &[f32],
    rows: usize,
    t: usize,
    r: usize,
    out: &mut [f32],
) {
    let cols = q.cols;
    let mut sums = [[0f32; C]; R];
    for block in 0..cols / 256 {
        #[cfg(target_arch = "wasm32")]
        let dots = crate::profile::measure("integer_dot", || unsafe {
            dot_tile::<R, C>(
                q.values.as_ptr().add(t * cols),
                weights.as_ptr().add(r * cols),
                cols,
                block * 256,
            )
        });
        #[cfg(not(target_arch = "wasm32"))]
        let dots = {
            let mut d = [[0i32; C]; R];
            for i in 0..R {
                for j in 0..C {
                    for c in block * 256..block * 256 + 256 {
                        d[i][j] +=
                            q.values[(t + i) * cols + c] as i32 * weights[(r + j) * cols + c] as i32;
                    }
                }
            }
            d
        };
        crate::profile::measure("integer_scale_sum", || {
            #[cfg(target_arch = "wasm32")]
            unsafe {
                scale_tile::<R, C>(
                    &dots,
                    &mut sums,
                    q.scales.as_ptr().add(t * (cols / 256) + block),
                    cols / 256,
                    scales.as_ptr().add(r),
                );
            }
            #[cfg(not(target_arch = "wasm32"))]
            for i in 0..R {
                for j in 0..C {
                    sums[i][j] += (dots[i][j] as f32 * q.scales[(t + i) * (cols / 256) + block])
                        * scales[r + j];
                }
            }
        });
    }
    for i in 0..R {
        if t + i < q.rows {
            for j in 0..C {
                out[(t + i) * rows + r + j] = sums[i][j];
            }
        }
    }
}
// All dimensions and padded scale rows have been checked by project().
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn scale_tile<const R: usize, const C: usize>(
    dots: &[[i32; C]; R],
    sums: &mut [[f32; C]; R],
    sx: *const f32,
    stride: usize,
    sw: *const f32,
) {
    use core::arch::wasm32::*;
    for i in 0..R {
        let activation = f32x4_splat(*sx.add(i * stride));
        for j in (0..C).step_by(4) {
            let d = v128_load(dots.as_ptr().cast::<i32>().add(i * C + j).cast());
            let w = v128_load(sw.add(j).cast());
            let value = f32x4_mul(f32x4_mul(f32x4_convert_i32x4(d), activation), w);
            let p = sums.as_mut_ptr().cast::<f32>().add(i * C + j);
            v128_store(p.cast(), f32x4_add(v128_load(p.cast()), value));
        }
    }
}
// The loop-unrolled load-sharing layout follows Laya's MIT-licensed integer tile.
// See docs/licenses/Laya-MIT.txt. Block256 scaling and pointer setup are Imajev-specific.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_tile<const R: usize, const C: usize>(
    q: *const i16,
    w: *const i8,
    cols: usize,
    start: usize,
) -> [[i32; C]; R] {
    use core::arch::wasm32::*;
    let mut acc = [[i32x4_splat(0); C]; R];
    let qp: [*const i16; R] = core::array::from_fn(|i| q.add(i * cols + start));
    let wp: [*const i8; C] = core::array::from_fn(|j| w.add(j * cols + start));
    for c in (0..256).step_by(256) {
        let weights: [[v128; 32]; C] = core::array::from_fn(|j| {
            core::array::from_fn(|g| i16x8_extend_low_i8x16(v128_load64_zero(wp[j].add(c + g * 8).cast())))
        });
        macro_rules! columns {($i:literal,$input:ident;$($j:literal),*)=>{$(if C>$j {
            acc[$i][$j]=i32x4_add(acc[$i][$j],i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[0],weights[$j][0]),i32x4_dot_i16x8($input[1],weights[$j][1])),i32x4_add(i32x4_dot_i16x8($input[2],weights[$j][2]),i32x4_dot_i16x8($input[3],weights[$j][3]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[4],weights[$j][4]),i32x4_dot_i16x8($input[5],weights[$j][5])),i32x4_add(i32x4_dot_i16x8($input[6],weights[$j][6]),i32x4_dot_i16x8($input[7],weights[$j][7])))),i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[8],weights[$j][8]),i32x4_dot_i16x8($input[9],weights[$j][9])),i32x4_add(i32x4_dot_i16x8($input[10],weights[$j][10]),i32x4_dot_i16x8($input[11],weights[$j][11]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[12],weights[$j][12]),i32x4_dot_i16x8($input[13],weights[$j][13])),i32x4_add(i32x4_dot_i16x8($input[14],weights[$j][14]),i32x4_dot_i16x8($input[15],weights[$j][15]))))),i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[16],weights[$j][16]),i32x4_dot_i16x8($input[17],weights[$j][17])),i32x4_add(i32x4_dot_i16x8($input[18],weights[$j][18]),i32x4_dot_i16x8($input[19],weights[$j][19]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[20],weights[$j][20]),i32x4_dot_i16x8($input[21],weights[$j][21])),i32x4_add(i32x4_dot_i16x8($input[22],weights[$j][22]),i32x4_dot_i16x8($input[23],weights[$j][23])))),i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[24],weights[$j][24]),i32x4_dot_i16x8($input[25],weights[$j][25])),i32x4_add(i32x4_dot_i16x8($input[26],weights[$j][26]),i32x4_dot_i16x8($input[27],weights[$j][27]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[28],weights[$j][28]),i32x4_dot_i16x8($input[29],weights[$j][29])),i32x4_add(i32x4_dot_i16x8($input[30],weights[$j][30]),i32x4_dot_i16x8($input[31],weights[$j][31])))))));
        })*};}
        macro_rules! rows {($($i:literal),*)=>{$(if R>$i {
            let input:[v128;32]=core::array::from_fn(|g| v128_load(qp[$i].add(c+g*8).cast()));
            columns!($i,input;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        })*};}
        rows!(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        );
    }
    let mut out = [[0i32; C]; R];
    for i in 0..R {
        for j in 0..C {
            let v = acc[i][j];
            let halves = i32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
            let total = i32x4_add(halves, i32x4_shuffle::<1, 0, 3, 2>(halves, halves));
            out[i][j] = i32x4_extract_lane::<0>(total);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fabricated_activation_buffers_are_rejected() {
        let mut q = quantize_rows(&vec![1.; 256], 1, 256).unwrap();
        q.values.pop();
        assert!(project(&q, &vec![1; 8 * 256], &[1.; 8], 8).is_err());
        q.values.push(128);
        assert!(project(&q, &vec![1; 8 * 256], &[1.; 8], 8).is_err());
        q.values[2047] = 0;
        q.scales.clear();
        assert!(project(&q, &vec![1; 8 * 256], &[1.; 8], 8).is_err());
    }
    #[test]
    fn tiled_integer_dots_match_scalar_for_token_tail() {
        let x = (0..11 * 256)
            .map(|i| (i as f32 * 0.073).sin() * 3.)
            .collect::<Vec<_>>();
        let q = quantize_rows(&x, 11, 256).unwrap();
        let w = (0..16 * 256)
            .map(|i| (i % 255) as i16 - 127)
            .map(|v| v as i8)
            .collect::<Vec<_>>();
        let scales = vec![0.017; 16];
        let y = project(&q, &w, &scales, 16).unwrap();
        for t in 0..11 {
            for r in 0..16 {
                let dot = (0..256)
                    .map(|c| q.values[t * 256 + c] as i32 * w[r * 256 + c] as i32)
                    .sum::<i32>();
                assert_eq!(y[t * 16 + r], (dot as f32 * q.scales[t]) * 0.017);
            }
        }
    }
}
