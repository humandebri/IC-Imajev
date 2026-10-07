//! Same finite predicate for every IEEE F32 bit pattern; no arithmetic on values.
pub fn all_finite(x:&[f32])->bool {
 #[cfg(target_arch="wasm32")]unsafe{return scan::<16>(x);}
 #[cfg(not(target_arch="wasm32"))]x.iter().all(|v|v.is_finite())
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn scan<const WIDTH:usize>(x:&[f32])->bool {
 use core::arch::wasm32::*;
 let mut i=0;let mask=i32x4_splat(0x7f800000);
 while i+WIDTH<=x.len(){
  let mut bad=i32x4_splat(0);
  for j in(0..WIDTH).step_by(4){
   let bits=v128_load(x.as_ptr().add(i+j).cast());
   bad=v128_or(bad,i32x4_eq(v128_and(bits,mask),mask));
  }
  if v128_any_true(bad){return false;}i+=WIDTH;
 }
 while i+4<=x.len(){if v128_any_true(i32x4_eq(v128_and(v128_load(x.as_ptr().add(i).cast()),mask),mask)){return false;}i+=4;}
 x[i..].iter().all(|v|v.is_finite())
}
