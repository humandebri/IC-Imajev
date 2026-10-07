//! Fused checked byte expansion with exact integer finite summaries.
type Result<T>=std::result::Result<T,String>;
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn finish_simd(
    low: core::arch::wasm32::v128,
    peak: core::arch::wasm32::v128,
) -> Result<bool> {
    use core::arch::wasm32::*;
    if v128_any_true(u32x4_ge(peak, u32x4_splat(0x7f800000))) {
        Err("invalid activation".into())
    } else {
        Ok(!v128_any_true(v128_and(low, i32x4_splat(65535))))
    }
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn bf16_8(bytes: &[u8], x: *mut f32, count: usize) -> Result<()> {
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
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn f32_4(bytes: &[u8], x: *mut f32, count: usize) -> Result<bool> {
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
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn bf16_64(bytes:&[u8],x:*mut f32,count:usize)->Result<()>{use core::arch::wasm32::*;let mut peak=i16x8_splat(0);let mask=i16x8_splat(0x7fff);let end=count/64*64;for i in(0..end).step_by(64){let a=v128_load(bytes.as_ptr().add((i+0)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+0).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+0+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+8)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+8).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+8+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+16)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+16).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+16+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+24)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+24).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+24+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+32)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+32).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+32+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+40)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+40).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+40+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+48)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+48).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+48+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));let a=v128_load(bytes.as_ptr().add((i+56)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+56).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+56+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));}let tail=count/8*8;for i in(end..tail).step_by(8){let a=v128_load(bytes.as_ptr().add((i+0)*2).cast());peak=u16x8_max(peak,v128_and(a,mask));v128_store(x.add(i+0).cast(),i32x4_shl(i32x4_extend_low_u16x8(a),16));v128_store(x.add(i+0+4).cast(),i32x4_shl(i32x4_extend_high_u16x8(a),16));}for i in tail..count{let bits=u16::from_le_bytes([bytes[i*2],bytes[i*2+1]]);peak=u16x8_max(peak,u16x8_splat(bits&0x7fff));x.add(i).write(f32::from_bits((bits as u32)<<16));}if v128_any_true(u16x8_ge(peak,u16x8_splat(0x7f80))){Err("invalid activation".into())}else{Ok(())}}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn f32_64(bytes:&[u8],x:*mut f32,count:usize)->Result<bool>{use core::arch::wasm32::*;let(mut low,mut peak)=(i32x4_splat(0),i32x4_splat(0));let mask=i32x4_splat(0x7fffffff);let end=count/64*64;for i in(0..end).step_by(64){let bits=v128_load(bytes.as_ptr().add((i+0)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+0).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+4)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+4).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+8)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+8).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+12)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+12).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+16)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+16).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+20)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+20).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+24)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+24).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+28)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+28).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+32)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+32).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+36)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+36).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+40)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+40).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+44)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+44).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+48)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+48).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+52)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+52).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+56)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+56).cast(),bits);let bits=v128_load(bytes.as_ptr().add((i+60)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+60).cast(),bits);}let tail=count/4*4;for i in(end..tail).step_by(4){let bits=v128_load(bytes.as_ptr().add((i+0)*4).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));v128_store(x.add(i+0).cast(),bits);}for i in tail..count{let bits=u32::from_le_bytes(bytes[i*4..i*4+4].try_into().unwrap());let lanes=i32x4_splat(bits as i32);low=v128_or(low,lanes);peak=u32x4_max(peak,v128_and(lanes,mask));x.add(i).write(f32::from_bits(bits));}finish_simd(low,peak)}
