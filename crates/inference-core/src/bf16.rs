//! Lossless all-BF16 fast path. Mixed F32 frames retain the generic bitmap codec.
pub fn all_bf16(x: &[f32]) -> bool {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        return all_simd(x);
    }
    #[cfg(not(target_arch = "wasm32"))]
    x.iter().all(|v| v.to_bits() & 0xffff == 0)
}
/// Write high BF16 bits. Caller must establish `all_bf16(x)` for lossless use.
/// Panics when the destination length is not exactly twice the input length.
pub fn pack(x: &[f32], bytes: &mut [u8]) {
    assert_eq!(bytes.len(), x.len() * 2);
    #[cfg(target_arch = "wasm32")]
    unsafe {
        pack_simd(x, bytes);
    }
    #[cfg(not(target_arch = "wasm32"))]
    for (v, b) in x.iter().zip(bytes.chunks_exact_mut(2)) {
        b.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
}
/// Expand little-endian BF16 bits exactly, including signed zero.
/// Panics when byte length is not exactly twice the output length.
pub fn unpack(bytes: &[u8], x: &mut [f32]) {
    assert_eq!(bytes.len(), x.len() * 2);
    #[cfg(target_arch = "wasm32")]
    unsafe {
        unpack_simd(bytes, x);
    }
    #[cfg(not(target_arch = "wasm32"))]
    for (b, v) in bytes.chunks_exact(2).zip(x) {
        *v = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
    }
}
/// Classify low BF16 bits and reject Inf/NaN in one complete pass.
pub fn classify_finite(x: &[f32]) -> crate::Result<bool> {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        return classify_simd(x);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (mut low, mut peak) = (0u32, 0u32);
        for v in x {
            let bits = v.to_bits();
            low |= bits;
            peak = peak.max(bits & 0x7fffffff);
        }
        finish_summary(low, peak)
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn finish_summary(low: u32, peak: u32) -> crate::Result<bool> {
    if peak >= 0x7f800000 {
        Err("invalid activation".into())
    } else {
        Ok(low & 65535 == 0)
    }
}
/// Decode checked BF16 bytes into spare capacity; no initialized value is read.
pub(crate) fn unpack_finite_uninit(
    bytes: &[u8],
    x: &mut [std::mem::MaybeUninit<f32>],
) -> crate::Result<()> {
    assert_eq!(bytes.len(), x.len() * 2);
    #[cfg(target_arch = "wasm32")]
    unsafe {
        return unpack_finite_simd(bytes, x.as_mut_ptr().cast(), x.len());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut peak = 0u16;
        for (b, v) in bytes.chunks_exact(2).zip(x) {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            peak = peak.max(bits & 0x7fff);
            v.write(f32::from_bits((bits as u32) << 16));
        }
        if peak >= 0x7f80 {
            Err("invalid activation".into())
        } else {
            Ok(())
        }
    }
}
/// Copy, finite-check and classify F32 bytes into spare capacity in one pass.
pub(crate) fn unpack_f32_finite_uninit(
    bytes: &[u8],
    x: &mut [std::mem::MaybeUninit<f32>],
) -> crate::Result<bool> {
    assert_eq!(bytes.len(), x.len() * 4);
    #[cfg(target_arch = "wasm32")]
    unsafe {
        return unpack_f32_finite_simd(bytes, x.as_mut_ptr().cast(), x.len());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (mut low, mut peak) = (0u32, 0u32);
        for (b, v) in bytes.chunks_exact(4).zip(x) {
            let bits = u32::from_le_bytes(b.try_into().unwrap());
            low |= bits;
            peak = peak.max(bits & 0x7fffffff);
            v.write(f32::from_bits(bits));
        }
        finish_summary(low, peak)
    }
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn finish_simd(
    low: core::arch::wasm32::v128,
    peak: core::arch::wasm32::v128,
) -> crate::Result<bool> {
    use core::arch::wasm32::*;
    if v128_any_true(u32x4_ge(peak, u32x4_splat(0x7f800000))) {
        Err("invalid activation".into())
    } else {
        Ok(!v128_any_true(v128_and(low, i32x4_splat(65535))))
    }
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn classify_simd(x: &[f32]) -> crate::Result<bool> {
    use core::arch::wasm32::*;
    let (mut low, mut peak) = (i32x4_splat(0), i32x4_splat(0));
    let mask = i32x4_splat(0x7fffffff);
    let end = x.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        let bits = v128_load(x.as_ptr().add(i).cast());
        low = v128_or(low, bits);
        peak = u32x4_max(peak, v128_and(bits, mask));
    }
    for v in &x[end..] {
        let bits = i32x4_splat(v.to_bits() as i32);
        low = v128_or(low, bits);
        peak = u32x4_max(peak, v128_and(bits, mask));
    }
    finish_simd(low, peak)
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn unpack_finite_simd(bytes: &[u8], x: *mut f32, count: usize) -> crate::Result<()> {
    use core::arch::wasm32::*;
    let mut peak = i16x8_splat(0);
    let mask = i16x8_splat(0x7fff);
    let end = count / 8 * 8;
    for i in (0..end).step_by(8) {
        let a = v128_load(bytes.as_ptr().add(i * 2).cast());
        peak = u16x8_max(peak, v128_and(a, mask));
        v128_store(x.add(i).cast(), i32x4_shl(i32x4_extend_low_u16x8(a), 16));
        v128_store(
            x.add(i + 4).cast(),
            i32x4_shl(i32x4_extend_high_u16x8(a), 16),
        );
    }
    for i in end..count {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        peak = u16x8_max(peak, u16x8_splat(bits & 0x7fff));
        x.add(i).write(f32::from_bits((bits as u32) << 16));
    }
    if v128_any_true(u16x8_ge(peak, u16x8_splat(0x7f80))) {
        Err("invalid activation".into())
    } else {
        Ok(())
    }
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn unpack_f32_finite_simd(bytes: &[u8], x: *mut f32, count: usize) -> crate::Result<bool> {
    use core::arch::wasm32::*;
    let (mut low, mut peak) = (i32x4_splat(0), i32x4_splat(0));
    let mask = i32x4_splat(0x7fffffff);
    let end = count / 4 * 4;
    for i in (0..end).step_by(4) {
        let bits = v128_load(bytes.as_ptr().add(i * 4).cast());
        low = v128_or(low, bits);
        peak = u32x4_max(peak, v128_and(bits, mask));
        v128_store(x.add(i).cast(), bits);
    }
    for i in end..count {
        let bits = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let lanes = i32x4_splat(bits as i32);
        low = v128_or(low, lanes);
        peak = u32x4_max(peak, v128_and(lanes, mask));
        x.add(i).write(f32::from_bits(bits));
    }
    finish_simd(low, peak)
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn all_simd(x: &[f32]) -> bool {
    use core::arch::wasm32::*;
    let mut low = i32x4_splat(0);
    let end = x.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        low = v128_or(low, v128_load(x.as_ptr().add(i).cast()));
    }
    !v128_any_true(v128_and(low, i32x4_splat(65535)))
        && x[end..].iter().all(|v| v.to_bits() & 65535 == 0)
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn pack_simd(x: &[f32], bytes: &mut [u8]) {
    use core::arch::wasm32::*;
    let end = x.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        let a = u32x4_shr(v128_load(x.as_ptr().add(i).cast()), 16);
        let b = u32x4_shr(v128_load(x.as_ptr().add(i + 4).cast()), 16);
        v128_store(
            bytes.as_mut_ptr().add(i * 2).cast(),
            u16x8_narrow_i32x4(a, b),
        );
    }
    for i in end..x.len() {
        let b = ((x[i].to_bits() >> 16) as u16).to_le_bytes();
        bytes[i * 2] = b[0];
        bytes[i * 2 + 1] = b[1];
    }
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn unpack_simd(bytes: &[u8], x: &mut [f32]) {
    use core::arch::wasm32::*;
    let end = x.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        let a = v128_load(bytes.as_ptr().add(i * 2).cast());
        v128_store(
            x.as_mut_ptr().add(i).cast(),
            i32x4_shl(i32x4_extend_low_u16x8(a), 16),
        );
        v128_store(
            x.as_mut_ptr().add(i + 4).cast(),
            i32x4_shl(i32x4_extend_high_u16x8(a), 16),
        );
    }
    for i in end..x.len() {
        x[i] = f32::from_bits((u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_bf16_bit_patterns_have_the_expected_finite_class() {
        for bits in 0..=u16::MAX {
            let v = f32::from_bits((bits as u32) << 16);
            assert_eq!(classify_finite(&[v]).is_ok(), v.is_finite());
        }
        for n in 0..35 {
            let mut x = vec![-0.; n];
            assert_eq!(classify_finite(&x).unwrap(), true);
            if n > 0 {
                x[n - 1] = f32::from_bits(1);
                assert_eq!(classify_finite(&x).unwrap(), false);
                x[n - 1] = f32::from_bits(0xff800001);
                assert!(classify_finite(&x).is_err());
            }
        }
    }
    #[test]
    fn pure_bf16_and_tails_preserve_signed_bits() {
        for n in 0..35 {
            let x: Vec<_> = (0..n)
                .map(|i| f32::from_bits(((i * 1739 + 0x8000) as u32) << 16))
                .collect();
            assert!(all_bf16(&x));
            let mut b = vec![0; n * 2];
            pack(&x, &mut b);
            let mut y = vec![0.; n];
            unpack(&b, &mut y);
            assert!(x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert!(!all_bf16(&[f32::from_bits(1)]));
    }
}
