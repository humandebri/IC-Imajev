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
    /// Put only the newly quantized block-aligned columns in their original
    /// positions. Other blocks are initialized but never projected again.
    #[cfg(feature="experimental-int8-k-continue")]
    pub(crate) fn place_columns(&self,cols:usize,begin:usize)->Result<Self> {
        if cols==0 || cols>262144 || cols%256!=0 || begin%256!=0
            || begin.checked_add(self.cols).is_none_or(|end|end>cols)
            || self.rows.checked_mul(cols).is_none_or(|size|size>crate::MAX_FLOATS) {
            return Err("continued integer column placement".into());
        }
        let padded=self.rows.div_ceil(8)*8;
        let mut values=vec![0;padded*cols];let mut scales=vec![1.;padded*(cols/256)];
        for token in 0..self.rows {
            values[token*cols+begin..token*cols+begin+self.cols].copy_from_slice(&self.values[token*self.cols..(token+1)*self.cols]);
            scales[token*(cols/256)+begin/256..token*(cols/256)+(begin+self.cols)/256].copy_from_slice(&self.scales[token*(self.cols/256)..(token+1)*(self.cols/256)]);
        }
        Ok(Self {strassen:std::cell::OnceCell::new(),values,scales,rows:self.rows,cols,
            #[cfg(feature="experimental-prepared-output-pairs")] output_pairs:std::cell::OnceCell::new()})
    }
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
                }?;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let v = &x[start..start + 256];
                let bits = v.iter().map(|v|v.to_bits() & 0x7fffffff).max().unwrap();
                if bits >= 0x7f800000 {return Err("integer projection input".into());}
                let max = f32::from_bits(bits);
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
// Canonical block256 projection; native code is the scalar numerical reference.
#[cfg(all(target_arch="wasm32",feature="experimental-paired-only"))]
pub fn project(_q:&QuantizedRows,_w:&[i8],_scales:&[f32],_rows:usize)->Result<Vec<f32>> {
    Err("fixed paired weights required for this build".into())
}
#[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
pub fn project(q:&QuantizedRows,w:&[i8],scales:&[f32],rows:usize)->Result<Vec<f32>> {
    let cols=q.cols;
    if rows==0 || rows%8!=0 || rows.checked_mul(cols)!=Some(w.len()) || w.len()>30_000_000
        || q.rows.checked_mul(rows).is_none_or(|v|v>crate::MAX_FLOATS)
        || scales.len()!=rows || !scales.iter().all(|v|v.is_finite()&&*v>0.) {
        return Err("integer projection weights".into());
    }
    let mut out=vec![0.;q.rows*rows];
    #[cfg(not(target_arch="wasm32"))]
    for token in 0..q.rows { for row in 0..rows { for block in 0..cols/256 {
        let mut dot=0i32;
        for c in block*256..(block+1)*256 {dot+=q.values[token*cols+c]as i32*w[row*cols+c]as i32;}
        out[token*rows+row]+=(dot as f32*q.scales[token*(cols/256)+block])*scales[row];
    } } }
    #[cfg(target_arch="wasm32")]
    for row in (0..rows).step_by(8) {
        let mut token=0;
        while token<q.rows {
            let remaining=if q.rows<8 {q.rows}else{q.rows.div_ceil(8)*8}-token;
            macro_rules! tile {($n:literal)=>{if remaining>=$n {project_tile::<$n>(q,w,scales,rows,token,row,&mut out);token+=$n;continue;}};}
            tile!(64);tile!(32);tile!(16);tile!(8);tile!(4);tile!(2);tile!(1);
        }
    }
    if !out.iter().all(|v|v.is_finite()) {return Err("integer projection output".into());}
    Ok(out)
}
#[cfg(all(target_arch="wasm32",not(feature="experimental-paired-only")))]
fn project_tile<const R:usize>(q:&QuantizedRows,w:&[i8],sw:&[f32],rows:usize,token:usize,row:usize,out:&mut[f32]) {
    let cols=q.cols;let mut sums=[[0.;8];R];
    for block in 0..cols/256 {
        crate::profile::measure("integer_dot_scale",||unsafe {
            crate::int8_tile::accumulate::<R>(q.values.as_ptr().add(token*cols),w.as_ptr().add(row*cols),cols,block*256,
                q.scales.as_ptr().add(token*(cols/256)+block),cols/256,sw.as_ptr().add(row),&mut sums);
        });
    }
    for i in 0..R {if token+i<q.rows {out[(token+i)*rows+row..(token+i)*rows+row+8].copy_from_slice(&sums[i]);}}
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
    fn peak_scan_matches_old_rounding_and_rejects_late_nonfinite() {
        for rows in [1,7,8,45,87,89,132] {
            let cols=512;let mut seed=0x8317a5u32;
            let x:Vec<f32>=(0..rows*cols).map(|_| {seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);f32::from_bits(seed % 0x7f800000 | (seed&0x80000000))}).collect();
            let q=quantize_rows(&x,rows,cols).unwrap();
            for (block,v) in x.chunks_exact(256).enumerate() {
                let peak=v.iter().map(|v|v.abs()).fold(0f32,f32::max);
                let scale=if peak==0.{1.}else{(peak/127.).max(f32::from_bits(1))};
                assert_eq!(q.scales()[block].to_bits(),scale.to_bits());
                for (i,&value) in v.iter().enumerate(){assert_eq!(q.values()[block*256+i],(value/scale).round_ties_even().clamp(-127.,127.)as i16);}
            }
        }
        for bits in [0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xff800001,0xffffffff] {
            for at in [0,255,256,511,1535] {let mut x=vec![1.;1536];x[at]=f32::from_bits(bits);assert!(quantize_rows(&x,3,512).is_err());}
        }
    }
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

#[cfg(test)]
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
