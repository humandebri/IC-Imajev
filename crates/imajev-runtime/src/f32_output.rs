//! Query-local output-lane SIMD with original F32 column accumulation order.
use crate::{prepared_weights::PreparedF32, Result};
pub(crate) fn project(
    x: &[f32],
    w: &PreparedF32,
    n: usize,
    rows: usize,
    cols: usize,
) -> Option<Result<Vec<f32>>> {
    let (packed, start, prepared_cols) = w.output_view()?;
    if n > 132 {
        return None;
    }
    Some((|| {
        if cols != prepared_cols
            || n == 0
            || rows == 0
            || n.checked_mul(cols) != Some(x.len())
            || rows.checked_mul(cols) != Some(w.len())
            || n.checked_mul(rows).is_none_or(|v| v > crate::MAX_FLOATS)
            || x.iter().any(|v| !v.is_finite())
        {
            return Err("output F32 projection shape/finite".into());
        }
        let mut out = vec![0.; n * rows];
        #[cfg(target_arch = "wasm32")]
        for r in (0..rows).step_by(32) {
            let width = (rows - r).min(32);
            let global = start + r;
            let pad;
            let weight = if width == 32 && global % 32 == 0 {
                &packed[global * cols..(global + 32) * cols]
            } else {
                pad = {
                    let mut values = vec![0.; 32 * cols];
                    for c in 0..cols {
                        for j in 0..width {
                            let row = global + j;
                            values[c * 32 + j] = packed[(row / 32) * 32 * cols + c * 32 + row % 32];
                        }
                    }
                    values
                };
                &pad
            };
            let mut sums = vec![[0f32; 32]; n];
            // Both private prepared constructors establish the complete fixed-weight
            // tile, and query validation establishes every real input token. The
            // patched body writes all n*32 initialized F32 output cells.
            for column in (0..cols).step_by(64) {
                unsafe {
                    accumulate(
                        x.as_ptr(),
                        weight.as_ptr(),
                        cols,
                        column,
                        x.as_ptr(),
                        32,
                        weight.as_ptr(),
                        sums.as_mut_ptr().cast(),
                        n,
                    );
                }
            }
            for t in 0..n {
                out[t * rows + r..t * rows + r + width].copy_from_slice(&sums[t][..width]);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        for t in 0..n {
            for r in 0..rows {
                let global = start + r;
                for c in 0..cols {
                    out[t * rows + r] +=
                        x[t * cols + c] * packed[(global / 32) * 32 * cols + c * 32 + global % 32];
                }
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err("output F32 projection finite".into());
        }
        Ok(out)
    })())
}
#[cfg(target_arch = "wasm32")]
#[export_name = "__imajev_f32_output64"]
#[inline(never)]
unsafe extern "C" fn accumulate(
    q: *const f32,
    w: *const f32,
    cols: usize,
    start: usize,
    sx: *const f32,
    stride: usize,
    sw: *const f32,
    out: *mut f32,
    n: usize,
) {
    let marker = core::hint::black_box(
        (q as usize)
            ^ (w as usize)
            ^ cols
            ^ start
            ^ (sx as usize)
            ^ stride
            ^ (sw as usize)
            ^ (out as usize)
            ^ n,
    ) as u32;
    for i in 0..n * 32 {
        core::ptr::write_volatile(out.add(i), f32::from_bits(marker | 0x7fc00000));
    }
}
