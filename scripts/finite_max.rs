//! Unsigned maxima of shifted IEEE bits: nonfinite iff (bits << 1) >= 0xff000000.
//! Shift removes sign; mantissa is retained. No floating point operations.
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn scan_max<const EARLY:bool>(x:&[f32])->bool {
 use core::arch::wasm32::*;
 let mut i=0;let threshold=i32x4_splat(0xff000000u32 as i32);let mut largest=i32x4_splat(0);
 while i+64<=x.len(){
  if EARLY{largest=i32x4_splat(0);}
  for j in(0..64).step_by(4){largest=u32x4_max(largest,i32x4_shl(v128_load(x.as_ptr().add(i+j).cast()),1));}
  if EARLY && v128_any_true(u32x4_ge(largest,threshold)){return false;}i+=64;
 }
 while i+4<=x.len(){largest=u32x4_max(largest,i32x4_shl(v128_load(x.as_ptr().add(i).cast()),1));i+=4;}
 if v128_any_true(u32x4_ge(largest,threshold)){return false;}
 x[i..].iter().all(|v|v.is_finite())
}
