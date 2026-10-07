//! A positive finite IEEE F32 has unsigned bits in 1..0x7f800000.
//! Subtracting one maps every invalid value at or above 0x7f7fffff.
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn scan_max<const EARLY:bool>(x:&[f32])->bool {
 use core::arch::wasm32::*;
 let mut i=0;let threshold=i32x4_splat(0x7f7fffff);let one=i32x4_splat(1);let mut largest=i32x4_splat(0);
 while i+64<=x.len(){
  if EARLY{largest=i32x4_splat(0);}
  for j in(0..64).step_by(4){largest=u32x4_max(largest,i32x4_sub(v128_load(x.as_ptr().add(i+j).cast()),one));}
  if EARLY && v128_any_true(u32x4_ge(largest,threshold)){return false;}i+=64;
 }
 while i+4<=x.len(){largest=u32x4_max(largest,i32x4_sub(v128_load(x.as_ptr().add(i).cast()),one));i+=4;}
 if v128_any_true(u32x4_ge(largest,threshold)){return false;}
 x[i..].iter().all(|v|v.is_finite()&&*v>0.)
}
pub fn all_positive_finite(x:&[f32])->bool {
 #[cfg(target_arch="wasm32")]unsafe{return scan_max::<false>(x);}
 #[cfg(not(target_arch="wasm32"))]x.iter().all(|v|v.is_finite()&&*v>0.)
}
