//! Exact high-bit BF16 packing via byte shuffle and unrolled decode.
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn pack8(x: &[f32], bytes: &mut [u8]) {
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
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn unpack8(bytes: &[u8], x: &mut [f32]) {
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
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn pack64(x:&[f32],bytes:&mut[u8]){use core::arch::wasm32::*;let end=x.len()/64*64;for i in(0..end).step_by(64){let a=v128_load(x.as_ptr().add(i+0).cast());let b=v128_load(x.as_ptr().add(i+0+4).cast());v128_store(bytes.as_mut_ptr().add((i+0)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));let a=v128_load(x.as_ptr().add(i+8).cast());let b=v128_load(x.as_ptr().add(i+8+4).cast());v128_store(bytes.as_mut_ptr().add((i+8)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));let a=v128_load(x.as_ptr().add(i+16).cast());let b=v128_load(x.as_ptr().add(i+16+4).cast());v128_store(bytes.as_mut_ptr().add((i+16)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));let a=v128_load(x.as_ptr().add(i+24).cast());let b=v128_load(x.as_ptr().add(i+24+4).cast());v128_store(bytes.as_mut_ptr().add((i+24)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));let a=v128_load(x.as_ptr().add(i+32).cast());let b=v128_load(x.as_ptr().add(i+32+4).cast());v128_store(bytes.as_mut_ptr().add((i+32)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));let a=v128_load(x.as_ptr().add(i+40).cast());let b=v128_load(x.as_ptr().add(i+40+4).cast());v128_store(bytes.as_mut_ptr().add((i+40)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));let a=v128_load(x.as_ptr().add(i+48).cast());let b=v128_load(x.as_ptr().add(i+48+4).cast());v128_store(bytes.as_mut_ptr().add((i+48)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));let a=v128_load(x.as_ptr().add(i+56).cast());let b=v128_load(x.as_ptr().add(i+56+4).cast());v128_store(bytes.as_mut_ptr().add((i+56)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));}let end8=x.len()/8*8;for i in(end..end8).step_by(8){let a=v128_load(x.as_ptr().add(i+0).cast());let b=v128_load(x.as_ptr().add(i+0+4).cast());v128_store(bytes.as_mut_ptr().add((i+0)*2).cast(),i8x16_shuffle::<2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31>(a,b));}for i in end8..x.len(){bytes[i*2..i*2+2].copy_from_slice(&((x[i].to_bits()>>16)as u16).to_le_bytes());}}

#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn unpack64(bytes:&[u8],x:&mut[f32]){use core::arch::wasm32::*;let end=x.len()/64*64;for i in(0..end).step_by(64){let a=v128_load(bytes.as_ptr().add((i+0)*2).cast());v128_store(x.as_mut_ptr().add(i+0).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+0+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+8)*2).cast());v128_store(x.as_mut_ptr().add(i+8).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+8+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+16)*2).cast());v128_store(x.as_mut_ptr().add(i+16).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+16+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+24)*2).cast());v128_store(x.as_mut_ptr().add(i+24).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+24+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+32)*2).cast());v128_store(x.as_mut_ptr().add(i+32).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+32+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+40)*2).cast());v128_store(x.as_mut_ptr().add(i+40).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+40+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+48)*2).cast());v128_store(x.as_mut_ptr().add(i+48).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+48+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+56)*2).cast());v128_store(x.as_mut_ptr().add(i+56).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+56+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));}let end8=x.len()/8*8;for i in(end..end8).step_by(8){let a=v128_load(bytes.as_ptr().add((i+0)*2).cast());v128_store(x.as_mut_ptr().add(i+0).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.as_mut_ptr().add(i+0+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));}for i in end8..x.len(){x[i]=f32::from_bits((u16::from_le_bytes([bytes[i*2],bytes[i*2+1]])as u32)<<16);}}
