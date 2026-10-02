//! Lossless all-BF16 fast path. Mixed F32 frames retain the generic bitmap codec.
pub fn all_bf16(x: &[f32]) -> bool {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        return all_simd(x);
    }
    #[cfg(not(target_arch = "wasm32"))]
    x.iter().all(|v| v.to_bits() & 0xffff == 0)
}
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
