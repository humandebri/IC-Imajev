//! Experimental W8/A8 projection: block256 activation scales, per-row weight scales.
//! Integer dots accumulate exactly in I32; scaling/summing blocks uses F32.
use crate::Result;
#[cfg(feature = "experimental-projection-reuse")]
use std::mem::MaybeUninit;
/// Validated, immutable block256 activations, created only by checked constructors.
///
/// ```compile_fail
/// let mut q = imajev_runtime::int8_kernel::quantize_rows(&vec![1.; 256], 1, 256).unwrap();
/// q.values.clear(); // Callers cannot invalidate the constructor invariants.
/// ```
#[derive(Debug)]
pub struct QuantizedRows {
    #[cfg(feature="experimental-strassen-raw")]
    strassen: std::cell::OnceCell<crate::output_pairs::strassen_raw::Operands>,
    values: Vec<i16>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
    #[cfg(feature="experimental-prepared-output-pairs")]
    output_pairs: std::cell::OnceCell<Vec<i16>>,
}
// Checked constructors and immutable accessors preserve the
// validated shape, finite positive scales, and signed [-127,127] lane bounds.
impl QuantizedRows {
    #[cfg(feature="experimental-strassen-raw")]
    pub(crate) fn strassen_operands(&self)->&crate::output_pairs::strassen_raw::Operands{self.strassen.get_or_init(||crate::output_pairs::strassen_raw::Operands::new(self))}
    #[cfg(feature = "experimental-projection-reuse")]
    pub(crate) fn from_bytes(rows: usize, cols: usize, values: &[u8], scales: &[f32]) -> Result<Self> {
        if rows == 0 || rows > 512 || cols == 0 || cols > 262144 || cols % 256 != 0
            || rows.checked_mul(cols) != Some(values.len())
            || rows.checked_mul(cols / 256) != Some(scales.len())
            || values.len() > crate::MAX_FLOATS
            || !scales.iter().all(|v| v.is_finite() && *v > 0.) {
            return Err("projection codec scales/shape".into());
        }
        let padded = rows.div_ceil(8) * 8;
        let total = padded * cols;
        let mut q = Vec::<i16>::with_capacity(total);
        // SAFETY: checked lengths are equal, divisible by256 and fit spare
        // capacity. Restoration only writes; an error leaves the Vec len0.
        #[cfg(target_arch = "wasm32")]
        unsafe { restore_bytes_simd(values,&mut q.spare_capacity_mut()[..values.len()])?; }
        #[cfg(not(target_arch = "wasm32"))]
        for (dst,v) in q.spare_capacity_mut().iter_mut().zip(values) {
            if *v == 128 { return Err("projection codec integer".into()); }
            dst.write((*v as i8) as i16);
        }
        // SAFETY: restoration succeeded and wrote every active element.
        // The helper initializes the remaining padding before committing len.
        unsafe { finish_rows(&mut q, values.len(), total); }
        let mut sx=vec![1.;padded*(cols/256)];
        sx[..scales.len()].copy_from_slice(scales);
        Ok(Self {#[cfg(feature="experimental-strassen-raw")] strassen:std::cell::OnceCell::new(),values:q,scales:sx,rows,cols, #[cfg(feature="experimental-prepared-output-pairs")] output_pairs:std::cell::OnceCell::new()})
    }
    #[cfg(feature = "experimental-projection-reuse")]
    pub(crate) fn from_wire(rows: usize, cols: usize, values: &[f32], scales: &[f32]) -> Result<Self> {
        if rows == 0 || rows > 512 || cols == 0 || cols > 262144 || cols % 256 != 0
            || rows.checked_mul(cols) != Some(values.len())
            || rows.checked_mul(cols / 256) != Some(scales.len())
            || !scales.iter().all(|v| v.is_finite() && *v > 0.) {
            return Err("prepared activation wire bounds".into());
        }
        let padded = rows.div_ceil(8) * 8;
        let total = padded * cols;
        let mut q = Vec::<i16>::with_capacity(total);
        // SAFETY: checked lengths are equal, divisible by256 and fit spare
        // capacity. Restoration only writes; an error leaves the Vec len0.
        #[cfg(target_arch = "wasm32")]
        unsafe { restore_wire(values, &mut q.spare_capacity_mut()[..values.len()])?; }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if !values.iter().all(|v| v.is_finite() && (-127. ..=127.).contains(v) && *v == (*v as i16) as f32) {
                return Err("prepared activation wire bounds".into());
            }
            for (dst, v) in q.spare_capacity_mut().iter_mut().zip(values) { dst.write(*v as i16); }
        }
        // SAFETY: restoration succeeded and wrote every active element.
        // The helper initializes the remaining padding before committing len.
        unsafe { finish_rows(&mut q, values.len(), total); }
        let mut sx = vec![1.; padded * (cols / 256)];
        sx[..scales.len()].copy_from_slice(scales);
        Ok(Self {#[cfg(feature="experimental-strassen-raw")] strassen:std::cell::OnceCell::new(), values: q, scales: sx, rows, cols, #[cfg(feature="experimental-prepared-output-pairs")] output_pairs:std::cell::OnceCell::new() })
    }
    #[cfg(feature = "experimental-projection-reuse")]
    pub(crate) fn append_wire_values(&self, out: &mut Vec<f32>) {
        let len = self.rows * self.cols;
        let start = out.len();
        out.reserve(len);
        // SAFETY: validated active rows have length divisible by256. The
        // matching spare slice is allocated and export writes every new slot.
        #[cfg(target_arch = "wasm32")]
        unsafe { export_wire(&self.values[..len], &mut out.spare_capacity_mut()[..len]); }
        #[cfg(not(target_arch = "wasm32"))]
        for (dst, v) in out.spare_capacity_mut()[..len].iter_mut().zip(&self.values[..len]) { dst.write(*v as f32); }
        // SAFETY: the old prefix was initialized and all appended slots were
        // written above. reserve proves capacity >= start+len.
        unsafe { out.set_len(start + len); }
    }
    pub fn values(&self) -> &[i16] {
        &self.values
    }
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[cfg(feature="experimental-prepared-output-pairs")]
    pub(crate) fn output_pair_values(&self)->&[i16] {
        self.output_pairs.get_or_init(||crate::output_pairs::duplicate(&self.values[..self.rows*self.cols]))
    }
}
#[cfg(all(target_arch = "wasm32", feature = "experimental-projection-reuse"))]
#[target_feature(enable = "simd128")]
unsafe fn restore_bytes_simd(values:&[u8],out:&mut[MaybeUninit<i16>])->Result<()> {
    use core::arch::wasm32::*;
    // The constructor proves equal lengths divisible by256. Destination
    // lanes are MaybeUninit: a failed late chunk never exposes earlier writes.
    // Each16-byte source load corresponds to two bounded8-lane I16 destination stores.
    for i in (0..values.len()).step_by(16) {
        let v=v128_load(values.as_ptr().add(i).cast());
        if i8x16_bitmask(i8x16_eq(v,i8x16_splat(-128)))!=0 {return Err("projection codec integer".into());}
        v128_store(out.as_mut_ptr().add(i).cast(),i16x8_extend_low_i8x16(v));
        v128_store(out.as_mut_ptr().add(i+8).cast(),i16x8_extend_high_i8x16(v));
    }
    Ok(())
}
#[cfg(all(target_arch = "wasm32", feature = "experimental-projection-reuse"))]
#[target_feature(enable = "simd128")]
unsafe fn restore_wire(input: &[f32], output: &mut [MaybeUninit<i16>]) -> Result<()> {
    use core::arch::wasm32::*;
    // The opaque constructor proves equal lengths divisible by 256. Verify
    // exact integer conversion and signed bounds before writing any SIMD lane.
    // Destination lanes are MaybeUninit; on error the owning Vec still has len0.
    for i in (0..input.len()).step_by(8) {
        let lo = v128_load(input.as_ptr().add(i).cast());
        let hi = v128_load(input.as_ptr().add(i + 4).cast());
        let q0 = i32x4_trunc_sat_f32x4(lo);
        let q1 = i32x4_trunc_sat_f32x4(hi);
        for (x,q) in [(lo,q0),(hi,q1)] {
            let exact = f32x4_eq(x, f32x4_convert_i32x4(q));
            let bounds = v128_and(i32x4_ge(q,i32x4_splat(-127)),i32x4_le(q,i32x4_splat(127)));
            if !i32x4_all_true(v128_and(exact,bounds)) { return Err("prepared activation wire bounds".into()); }
        }
        v128_store(output.as_mut_ptr().add(i).cast(),i16x8_narrow_i32x4(q0,q1));
    }
    Ok(())
}
#[cfg(all(target_arch = "wasm32", feature = "experimental-projection-reuse"))]
#[target_feature(enable = "simd128")]
unsafe fn export_wire(input: &[i16], output: &mut [MaybeUninit<f32>]) {
    use core::arch::wasm32::*;
    for i in (0..input.len()).step_by(8) {
        let q = v128_load(input.as_ptr().add(i).cast());
        v128_store(output.as_mut_ptr().add(i).cast(),f32x4_convert_i32x4(i32x4_extend_low_i16x8(q)));
        v128_store(output.as_mut_ptr().add(i+4).cast(),f32x4_convert_i32x4(i32x4_extend_high_i16x8(q)));
    }
}
/// Commit a freshly allocated row buffer after writing its active prefix.
///
/// # Safety
/// q must have len0 and capacity >= total, active <= total, and every element
/// in 0..active must already have been initialized. No destination references
/// may remain live. On constructor errors, this helper is never called.
unsafe fn finish_rows(q: &mut Vec<i16>, active: usize, total: usize) {
    debug_assert!(q.is_empty() && active <= total && total <= q.capacity());
    for slot in &mut q.spare_capacity_mut()[active..total] { slot.write(0); }
    // SAFETY: the caller initialized the active prefix, and the loop above
    // initialized every padding slot. i16 has no invalid bit patterns.
    unsafe { q.set_len(total); }
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
    let total = padded * cols;
    let mut q = Vec::<i16>::with_capacity(total);
    let mut scales = vec![1f32; padded * (cols / 256)];
    for r in 0..rows {
        for block in 0..cols / 256 {
            let start = r * cols + block * 256;
            #[cfg(target_arch = "wasm32")]
            {
                // SAFETY: this block spans 256 checked input elements and
                // 256 allocated destination slots. block writes every slot;
                // no initialized destination reference exists before completion.
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
                    // SAFETY: start+i is within the checked active prefix and
                    // capacity. Every active slot is visited exactly once.
                    unsafe { q.as_mut_ptr().add(start+i).write(
                        (v[i] / scale).round_ties_even().clamp(-127., 127.) as i16); }
                }
            }
        }
    }
    // SAFETY: each block initialized all 256 slots in the active prefix.
    // Initialize the token padding before exposing the immutable rows.
    unsafe { finish_rows(&mut q, x.len(), total); }
    Ok(QuantizedRows {
        #[cfg(feature="experimental-prepared-output-pairs")]
        output_pairs:std::cell::OnceCell::new(),
        #[cfg(feature="experimental-strassen-raw")]
        strassen:std::cell::OnceCell::new(),
        values: q,
        scales,
        rows,
        cols,
    })
}
#[cfg(all(target_arch="wasm32",feature="experimental-dot-scale"))]
#[path="int8_dot_scale.rs"]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
mod dot_scale;
#[cfg(all(target_arch="wasm32",feature="experimental-column32"))]
#[path="int8_column32.rs"]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
mod column32;
/// Fixed-pair deployments reject unprepared legacy projections. This removes
/// unused large fallback specializations from the Wasm build. The native and
/// default runtime continue to support those paths for comparison and tests.
#[cfg(all(target_arch="wasm32",feature="experimental-paired-only"))]
pub fn project(_q:&QuantizedRows,_w:&[i8],_scales:&[f32],_rows:usize)->Result<Vec<f32>> {
    Err("fixed paired weights required for this build".into())
}
/// Diagnostic: share each input load across 32 outputs for padded88 tokens.
/// All other shapes retain the adopted column16/balanced44 implementation.
#[cfg(feature="experimental-column32")]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
pub fn project_column32_balanced(q:&QuantizedRows,w:&[i8],scales:&[f32],rows:usize)->Result<Vec<f32>> {
    if q.rows.div_ceil(8)!=11 || rows%32!=0 {return project_balanced44(q,w,scales,rows);}
    let cols=q.cols;
    if rows==0 || rows.checked_mul(cols)!=Some(w.len()) || w.len()>30_000_000
        || q.rows.checked_mul(rows).is_none_or(|v|v>crate::MAX_FLOATS)
        || scales.len()!=rows || !scales.iter().all(|v|v.is_finite()&&*v>0.) {
        return Err("column32 weights".into());
    }
    let mut out=vec![0.;q.rows*rows];
    for r in (0..rows).step_by(32) {
        for t in [0,44] {
            let mut sums=[[0f32;32];44];
            for block in 0..cols/256 {
                #[cfg(target_arch="wasm32")]
                crate::profile::measure("integer_dot_scale",||unsafe {
                    column32::accumulate::<44,32>(q.values.as_ptr().add(t*cols),w.as_ptr().add(r*cols),cols,block*256,q.scales.as_ptr().add(t*(cols/256)+block),cols/256,scales.as_ptr().add(r),&mut sums)
                });
                #[cfg(not(target_arch="wasm32"))]
                for i in 0..44 {for j in 0..32 {
                    let mut d=0i32;
                    for c in block*256..(block+1)*256 {d+=q.values[(t+i)*cols+c]as i32*w[(r+j)*cols+c]as i32;}
                    sums[i][j]+=(d as f32*q.scales[(t+i)*(cols/256)+block])*scales[r+j];
                }}
            }
            for i in 0..44 {if t+i<q.rows {out[(t+i)*rows+r..(t+i)*rows+r+32].copy_from_slice(&sums[i]);}}
        }
    }
    if !out.iter().all(|v|v.is_finite()) {return Err("column32 output".into());}
    Ok(out)
}
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
pub fn project(q: &QuantizedRows, w: &[i8], scales: &[f32], rows: usize) -> Result<Vec<f32>> {
    #[cfg(feature="experimental-adopt-column16")]
    if rows%16==0 {
        #[cfg(feature="experimental-adopt-column32")]
        if q.rows.div_ceil(8)==11 && rows%32==0 {return project_column32_balanced(q,w,scales,rows);}
        #[cfg(feature="experimental-adopt-balanced44")]
        if q.rows.div_ceil(8)==11 {return project_balanced44(q,w,scales,rows);}
        #[cfg(feature="experimental-adopt-column16-token48")]
        return project_impl::<16, true, {cfg!(feature="experimental-adopt-dot-scale")},false>(q,w,scales,rows);
        #[cfg(not(feature="experimental-adopt-column16-token48"))]
        return project_impl::<16, false, {cfg!(feature="experimental-adopt-dot-scale")},false>(q,w,scales,rows);
    }
    project_impl::<8, false, {cfg!(feature="experimental-adopt-dot-scale")},false>(q,w,scales,rows)
}
/// Diagnostic alternative: share each input load across16 output columns.
#[cfg(feature="experimental-column16")]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
pub fn project_column16(q: &QuantizedRows, w: &[i8], scales: &[f32], rows: usize) -> Result<Vec<f32>> {
    project_impl::<16, false, false,false>(q,w,scales,rows)
}
#[cfg(feature="experimental-column16-token48")]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
pub fn project_column16_token48(q: &QuantizedRows, w: &[i8], scales: &[f32], rows: usize) -> Result<Vec<f32>> {
    project_impl::<16, true, false,false>(q,w,scales,rows)
}
#[cfg(feature="experimental-dot-scale")]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
pub fn project_dot_scale(q:&QuantizedRows,w:&[i8],scales:&[f32],rows:usize)->Result<Vec<f32>> {project_impl::<16,true,true,false>(q,w,scales,rows)}
#[cfg(feature="experimental-balanced44")]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
pub fn project_balanced44(q:&QuantizedRows,w:&[i8],scales:&[f32],rows:usize)->Result<Vec<f32>> {
    if q.rows.div_ceil(8)==11 {project_impl::<16,true,true,true>(q,w,scales,rows)}else{project_dot_scale(q,w,scales,rows)}
}
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
fn project_impl<const C:usize, const TOKEN48:bool, const FUSED:bool, const BALANCED:bool>(q: &QuantizedRows, w: &[i8], scales: &[f32], rows: usize) -> Result<Vec<f32>> {
    const {assert!(C==8 || C==16); }
    let cols = q.cols;
    // QuantizedRows is opaque and cannot be mutated or constructed by callers.
    // Reuse its constructor invariants instead of scanning every activation
    // again for each projection (including both halves of fused gate/up).
    if rows == 0
        || rows % C != 0
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
            let padded = if q.rows < 8 {
                q.rows
            } else {
                q.rows.div_ceil(8) * 8
            };
            // Exact padded token counts from the real five graphs. Handle the
            // whole group once so each block's fixed weights are expanded once,
            // rather than once again for each 64/32/16/8-token remainder.
            #[cfg(feature = "experimental-wide-token-tiles")]
            {
                macro_rules! whole {
                    ($r:literal) => {
                        if t == 0 && C==8 && padded == $r {
                            project_tile::<$r, 8,FUSED>(q, weights, scales, rows, 0, r, &mut out);
                            t += $r;
                            continue;
                        }
                    };
                }
                whole!(48);
                whole!(80);
                whole!(88);
                whole!(96);
                whole!(136);
            }
            // Eight-token padding is unchanged. Cover 88 rows with two
            // immediate fused groups, avoiding the separate delayed R8 tail.
            #[cfg(feature="experimental-balanced44")]
            if BALANCED && C==16 && TOKEN48 && padded==88 {
                project_tile::<44,16,FUSED>(q,weights,scales,rows,t,r,&mut out);
                t+=44;
                continue;
            }
            // Reuse each expanded weight across 48 tokens. The original block
            // scaling and each token's accumulation order remain unchanged.
            #[cfg(feature="experimental-column16-token48")]
            if TOKEN48 && C==16 && t+48<=padded {
                project_tile::<48,16,FUSED>(q,weights,scales,rows,t,r,&mut out);
                t+=48;
                continue;
            }
            macro_rules! dispatch {
                ($c:ident) => {
                    if C==8 && t + 64 <= padded {
                        project_tile::<64, $c,FUSED>(q, &weights, scales, rows, t, r, &mut out);
                        t += 64;
                    } else if t + 32 <= padded {
                        project_tile::<32, $c,FUSED>(q, &weights, scales, rows, t, r, &mut out);
                        t += 32;
                    } else if t + 16 <= padded {
                        project_tile::<16, $c,FUSED>(q, &weights, scales, rows, t, r, &mut out);
                        t += 16;
                    } else if t + 8 <= padded {
                        project_tile::<8, $c,FUSED>(q, &weights, scales, rows, t, r, &mut out);
                        t += 8;
                    } else if t + 4 <= padded {
                        project_tile::<4, $c,FUSED>(q, &weights, scales, rows, t, r, &mut out);
                        t += 4;
                    } else if t + 2 <= padded {
                        project_tile::<2, $c,FUSED>(q, &weights, scales, rows, t, r, &mut out);
                        t += 2;
                    } else {
                        project_tile::<1, $c,FUSED>(q, &weights, scales, rows, t, r, &mut out);
                        t += 1;
                    }
                };
            }
            dispatch!(C);
        }
        r += C;
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err("integer projection output".into());
    }
    Ok(out)
}
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
fn project_tile<const R: usize, const C: usize, const FUSED:bool>(
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
        #[cfg(all(target_arch="wasm32",feature="experimental-dot-scale"))]
        if FUSED && R>8 && R<=64 {
            // Same integer reduction and (dot*sx)*sw+sum; no temporary dot array.
            crate::profile::measure("integer_dot_scale",||unsafe {dot_scale::accumulate::<R,C>(q.values.as_ptr().add(t*cols),weights.as_ptr().add(r*cols),cols,block*256,q.scales.as_ptr().add(t*(cols/256)+block),cols/256,scales.as_ptr().add(r),&mut sums)});
            continue;
        }
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
                        d[i][j] += q.values[(t + i) * cols + c] as i32
                            * weights[(r + j) * cols + c] as i32;
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
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
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
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
unsafe fn dot_tile<const R: usize, const C: usize>(
    q: *const i16,
    w: *const i8,
    cols: usize,
    start: usize,
) -> [[i32; C]; R] {
    #[cfg(feature = "experimental-wide-token-tiles")]
    if R > 64 {
        let mut out = [[0i32; C]; R];
        dot_tile_wide::<C>(q, w, cols, start, R, out.as_mut_ptr().cast());
        return out;
    }
    if R <= 8 {
        dot_tile_delayed::<R, C>(q, w, cols, start)
    } else {
        dot_tile_immediate::<R, C>(q, w, cols, start)
    }
}
#[cfg(all(target_arch = "wasm32", feature = "experimental-wide-token-tiles"))]
#[target_feature(enable = "simd128")]
#[inline(never)]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
unsafe fn dot_tile_wide<const C: usize>(
    q: *const i16, w: *const i8, cols: usize, start: usize,
    tokens: usize, out: *mut i32,
) {
    const { assert!(C <= 16 && C % 4 == 0); }
    use core::arch::wasm32::*;
    // The caller supplies exactly R padded rows and an R*C output allocation.
    // Keep the coefficient block live across every token row.
    let weights: [[v128; 32]; C] = core::array::from_fn(|j| {
        core::array::from_fn(|g| i16x8_extend_low_i8x16(
            v128_load64_zero(w.add(j * cols + start + g * 8).cast())))
    });
        macro_rules! dot {
            ($input:ident,$j:expr) => {
                i32x4_add(
                    i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[0], weights[$j][0]),
                                    i32x4_dot_i16x8($input[1], weights[$j][1]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[2], weights[$j][2]),
                                    i32x4_dot_i16x8($input[3], weights[$j][3]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[4], weights[$j][4]),
                                    i32x4_dot_i16x8($input[5], weights[$j][5]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[6], weights[$j][6]),
                                    i32x4_dot_i16x8($input[7], weights[$j][7]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[8], weights[$j][8]),
                                    i32x4_dot_i16x8($input[9], weights[$j][9]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[10], weights[$j][10]),
                                    i32x4_dot_i16x8($input[11], weights[$j][11]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[12], weights[$j][12]),
                                    i32x4_dot_i16x8($input[13], weights[$j][13]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[14], weights[$j][14]),
                                    i32x4_dot_i16x8($input[15], weights[$j][15]),
                                ),
                            ),
                        ),
                    ),
                    i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[16], weights[$j][16]),
                                    i32x4_dot_i16x8($input[17], weights[$j][17]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[18], weights[$j][18]),
                                    i32x4_dot_i16x8($input[19], weights[$j][19]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[20], weights[$j][20]),
                                    i32x4_dot_i16x8($input[21], weights[$j][21]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[22], weights[$j][22]),
                                    i32x4_dot_i16x8($input[23], weights[$j][23]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[24], weights[$j][24]),
                                    i32x4_dot_i16x8($input[25], weights[$j][25]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[26], weights[$j][26]),
                                    i32x4_dot_i16x8($input[27], weights[$j][27]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[28], weights[$j][28]),
                                    i32x4_dot_i16x8($input[29], weights[$j][29]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[30], weights[$j][30]),
                                    i32x4_dot_i16x8($input[31], weights[$j][31]),
                                ),
                            ),
                        ),
                    ),
                )
            };
        }
        macro_rules! group {
            ($i:expr,$input:ident,$j:literal) => {
                if C >= $j + 4 {
                    let a0 = dot!($input, $j);
                    let a1 = dot!($input, { $j + 1 });
                    let a2 = dot!($input, { $j + 2 });
                    let a3 = dot!($input, { $j + 3 });
                    let a = i32x4_add(
                        i32x4_shuffle::<0, 1, 4, 5>(a0, a1),
                        i32x4_shuffle::<2, 3, 6, 7>(a0, a1),
                    );
                    let b = i32x4_add(
                        i32x4_shuffle::<0, 1, 4, 5>(a2, a3),
                        i32x4_shuffle::<2, 3, 6, 7>(a2, a3),
                    );
                    let total = i32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(a, b),
                        i32x4_shuffle::<1, 3, 5, 7>(a, b),
                    );
                    v128_store(out.add($i * C + $j).cast(), total);
                }
            };
        }
        macro_rules! columns {
            ($i:expr,$input:ident;$($unused:literal),*) => {
                group!($i, $input, 0);
                group!($i, $input, 4);
                group!($i, $input, 8);
                group!($i, $input, 12);
            };
        }

    for i in 0..tokens {
        let input: [v128; 32] = core::array::from_fn(|g| v128_load(q.add(i * cols + start + g * 8).cast()));
        columns!(i, input; 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
unsafe fn dot_tile_immediate<const R: usize, const C: usize>(
    q: *const i16,
    w: *const i8,
    cols: usize,
    start: usize,
) -> [[i32; C]; R] {
    assert!(R <= 64);
    const { assert!(C <= 16 && C % 4 == 0); }
    use core::arch::wasm32::*;
    let mut out = [[0i32; C]; R];
    let qp: [*const i16; R] = core::array::from_fn(|i| q.add(i * cols + start));
    let wp: [*const i8; C] = core::array::from_fn(|j| w.add(j * cols + start));
    for c in (0..256).step_by(256) {
        #[cfg(not(feature="experimental-explicit-weight-loads"))]
        let weights: [[v128; 32]; C] = core::array::from_fn(|j| {
            core::array::from_fn(|g| {
                i16x8_extend_low_i8x16(v128_load64_zero(wp[j].add(c + g * 8).cast()))
            })
        });
        #[cfg(feature="experimental-explicit-weight-loads")]
        let weights: [[v128;32];C] = {
            // All loads are within the caller's C checked weight rows and
            // their block256. Explicit columns prevent an outer from_fn loop
            // from materializing the entire expanded weight array in memory.
            let mut weights=[[i32x4_splat(0);32];C];
            macro_rules! column {($j:literal)=>{if C>$j {weights[$j]=[
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 0).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 8).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 16).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 24).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 32).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 40).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 48).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 56).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 64).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 72).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 80).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 88).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 96).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 104).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 112).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 120).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 128).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 136).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 144).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 152).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 160).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 168).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 176).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 184).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 192).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 200).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 208).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 216).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 224).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 232).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 240).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 248).cast()))
            ];}};}
            column!(0);column!(1);column!(2);column!(3);
            column!(4);column!(5);column!(6);column!(7);
            column!(8);column!(9);column!(10);column!(11);
            column!(12);column!(13);column!(14);column!(15);
            weights
        };
        macro_rules! dot {
            ($input:ident,$j:expr) => {
                i32x4_add(
                    i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[0], weights[$j][0]),
                                    i32x4_dot_i16x8($input[1], weights[$j][1]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[2], weights[$j][2]),
                                    i32x4_dot_i16x8($input[3], weights[$j][3]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[4], weights[$j][4]),
                                    i32x4_dot_i16x8($input[5], weights[$j][5]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[6], weights[$j][6]),
                                    i32x4_dot_i16x8($input[7], weights[$j][7]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[8], weights[$j][8]),
                                    i32x4_dot_i16x8($input[9], weights[$j][9]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[10], weights[$j][10]),
                                    i32x4_dot_i16x8($input[11], weights[$j][11]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[12], weights[$j][12]),
                                    i32x4_dot_i16x8($input[13], weights[$j][13]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[14], weights[$j][14]),
                                    i32x4_dot_i16x8($input[15], weights[$j][15]),
                                ),
                            ),
                        ),
                    ),
                    i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[16], weights[$j][16]),
                                    i32x4_dot_i16x8($input[17], weights[$j][17]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[18], weights[$j][18]),
                                    i32x4_dot_i16x8($input[19], weights[$j][19]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[20], weights[$j][20]),
                                    i32x4_dot_i16x8($input[21], weights[$j][21]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[22], weights[$j][22]),
                                    i32x4_dot_i16x8($input[23], weights[$j][23]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[24], weights[$j][24]),
                                    i32x4_dot_i16x8($input[25], weights[$j][25]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[26], weights[$j][26]),
                                    i32x4_dot_i16x8($input[27], weights[$j][27]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[28], weights[$j][28]),
                                    i32x4_dot_i16x8($input[29], weights[$j][29]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[30], weights[$j][30]),
                                    i32x4_dot_i16x8($input[31], weights[$j][31]),
                                ),
                            ),
                        ),
                    ),
                )
            };
        }
        macro_rules! group {
            ($i:literal,$input:ident,$j:literal) => {
                if C >= $j + 4 {
                    let a0 = dot!($input, $j);
                    let a1 = dot!($input, { $j + 1 });
                    let a2 = dot!($input, { $j + 2 });
                    let a3 = dot!($input, { $j + 3 });
                    let a = i32x4_add(
                        i32x4_shuffle::<0, 1, 4, 5>(a0, a1),
                        i32x4_shuffle::<2, 3, 6, 7>(a0, a1),
                    );
                    let b = i32x4_add(
                        i32x4_shuffle::<0, 1, 4, 5>(a2, a3),
                        i32x4_shuffle::<2, 3, 6, 7>(a2, a3),
                    );
                    let total = i32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(a, b),
                        i32x4_shuffle::<1, 3, 5, 7>(a, b),
                    );
                    v128_store(out[$i].as_mut_ptr().add($j).cast(), total);
                }
            };
        }
        macro_rules! columns {
            ($i:literal,$input:ident;$($unused:literal),*) => {
                group!($i, $input, 0);
                group!($i, $input, 4);
                group!($i, $input, 8);
                group!($i, $input, 12);
            };
        }
        macro_rules! rows {($($i:literal),*)=>{$(if R>$i {
            let input:[v128;32]=core::array::from_fn(|g| v128_load(qp[$i].add(c+g*8).cast()));
            columns!($i,input;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        })*};}
        rows!(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        );
    }
    out
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
unsafe fn dot_tile_delayed<const R: usize, const C: usize>(
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
            core::array::from_fn(|g| {
                i16x8_extend_low_i8x16(v128_load64_zero(wp[j].add(c + g * 8).cast()))
            })
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
    macro_rules! store_cols {($i:literal;$($j:literal),*)=>{$(if C>$j {
        let v=acc[$i][$j];
        let halves=i32x4_add(v,i32x4_shuffle::<2,3,0,1>(v,v));
        let total=i32x4_add(halves,i32x4_shuffle::<1,0,3,2>(halves,halves));
        out[$i][$j]=i32x4_extract_lane::<0>(total);
    })*};}
    macro_rules! store_rows {($($i:literal),*)=>{$(if R>$i {store_cols!($i;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);})*};}
    store_rows!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_projection_weights_are_still_rejected() {
        let q = quantize_rows(&vec![1.; 256], 1, 256).unwrap();
        assert!(project(&q, &vec![1; 8 * 256 - 1], &[1.; 8], 8).is_err());
        assert!(project(&q, &vec![1; 8 * 256], &[1.; 7], 8).is_err());
        for bad in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(project(&q, &vec![1; 8 * 256], &[bad; 8], 8).is_err());
        }
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

#[cfg(test)]
mod invariant_tests {
    use super::*;
    #[test]
    fn constructor_preserves_extreme_and_padded_activation_invariants() {
        for rows in [1, 7, 8, 9] {
            let mut x = vec![0.; rows * 256];
            for (i, v) in x.iter_mut().enumerate() {
                *v = [0., f32::from_bits(1), -f32::from_bits(1), f32::MAX, -f32::MAX][i % 5];
            }
            let q = quantize_rows(&x, rows, 256).unwrap();
            assert_eq!(q.values().len(), rows.div_ceil(8) * 8 * 256);
            assert!(q.values().iter().all(|v| (-127..=127).contains(v)));
            assert!(q.scales().iter().all(|v| v.is_finite() && *v > 0.));
            assert!(q.values()[rows * 256..].iter().all(|v| *v == 0));
            assert!(q.scales()[rows..].iter().all(|v| *v == 1.));
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(quantize_rows(&vec![bad; 256], 1, 256).is_err());
        }
    }
}

#[cfg(all(test, feature = "experimental-wide-token-tiles"))]
mod whole_token_tests {
    use super::*;
    #[test]
    fn whole_real_token_groups_preserve_two_block_scaling_and_tail() {
        for tokens in [45, 80, 87, 89, 132] {
            let cols = 512;
            let x: Vec<_> = (0..tokens * cols).map(|i| (i as f32 * 0.013).sin() * 3.).collect();
            let q = quantize_rows(&x, tokens, cols).unwrap();
            let w: Vec<_> = (0..16 * cols).map(|i| (i % 255) as i16 - 127).map(|v| v as i8).collect();
            let scales: Vec<_> = (0..16).map(|i| 0.007 * (i + 1) as f32).collect();
            let got = project(&q, &w, &scales, 16).unwrap();
            for t in 0..tokens {
                for r in 0..16 {
                    let mut expected = 0f32;
                    for block in 0..2 {
                        let dot: i32 = (block * 256..(block + 1) * 256).map(|c| q.values[t * cols + c] as i32 * w[r * cols + c] as i32).sum();
                        expected += (dot as f32 * q.scales[t * 2 + block]) * scales[r];
                    }
                    assert_eq!(got[t * 16 + r].to_bits(), expected.to_bits(), "{tokens}/{t}/{r}");
                }
            }
        }
    }
}

#[cfg(all(test, feature = "experimental-projection-reuse"))]
mod initialization_tests {
    use super::*;
    #[test]
    fn restored_and_exported_rows_initialize_active_prefix_and_padding() {
        for rows in [1,7,8,31,87,132] {
            let cols=512;
            let values:Vec<_>=(0..rows*cols).map(|i| [-127i16,0,127,-1,1][i%5]).collect();
            let bytes:Vec<_>=values.iter().map(|v|*v as i8 as u8).collect();
            let floats:Vec<_>=values.iter().map(|v|*v as f32).collect();
            let scales=vec![0.0123;rows*(cols/256)];
            for q in [QuantizedRows::from_bytes(rows,cols,&bytes,&scales).unwrap(),
                      QuantizedRows::from_wire(rows,cols,&floats,&scales).unwrap()] {
                assert_eq!(&q.values()[..values.len()],&values);
                assert!(q.values()[values.len()..].iter().all(|v|*v==0));
                assert_eq!(&q.scales()[..scales.len()],&scales);
                assert!(q.scales()[scales.len()..].iter().all(|v|*v==1.));
                let mut out=vec![-0.,f32::from_bits(1),17.];
                q.append_wire_values(&mut out);
                q.append_wire_values(&mut out);
                assert_eq!(out[0].to_bits(),(-0f32).to_bits());
                assert_eq!(out[1].to_bits(),1);assert_eq!(out[2],17.);
                assert_eq!(&out[3..3+floats.len()],&floats);
                assert_eq!(&out[3+floats.len()..],&floats);
            }
        }
    }
    #[test]
    fn rejected_late_values_never_expose_partly_initialized_rows() {
        let rows=7;let cols=256;let scales=vec![1.;rows];
        for index in [0,rows*cols-1] {
            let mut bytes=vec![127u8;rows*cols];bytes[index]=128;
            assert!(QuantizedRows::from_bytes(rows,cols,&bytes,&scales).is_err());
            for bad in [0.5,-128.,128.,f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {
                let mut floats=vec![127.;rows*cols];floats[index]=bad;
                assert!(QuantizedRows::from_wire(rows,cols,&floats,&scales).is_err());
            }
        }
        // A successful constructor after failures must still initialize padding.
        let q=QuantizedRows::from_bytes(rows,cols,&vec![129;rows*cols],&scales).unwrap();
        assert!(q.values()[..rows*cols].iter().all(|v|*v == -127));
        assert!(q.values()[rows*cols..].iter().all(|v|*v==0));
    }
}
