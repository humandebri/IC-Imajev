//! BF16 predicates with identical integer summaries and fewer loop branches.
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn finish(low:core::arch::wasm32::v128,peak:core::arch::wasm32::v128)->Result<bool,String>{
 use core::arch::wasm32::*;
 if v128_any_true(u32x4_ge(peak,u32x4_splat(0x7f800000))){Err("invalid activation".into())}
 else{Ok(!v128_any_true(v128_and(low,i32x4_splat(65535))))}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn classify4(x:&[f32])->Result<bool,String>{
 use core::arch::wasm32::*;
 let(mut low,mut peak)=(i32x4_splat(0),i32x4_splat(0));let mask=i32x4_splat(0x7fffffff);let end=x.len()/4*4;
 for i in(0..end).step_by(4){let bits=v128_load(x.as_ptr().add(i).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));}
 for v in &x[end..]{let bits=i32x4_splat(v.to_bits()as i32);low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));}
 finish(low,peak)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn all4(x:&[f32])->bool{
 use core::arch::wasm32::*;
 let mut low=i32x4_splat(0);let end=x.len()/4*4;
 for i in(0..end).step_by(4){low=v128_or(low,v128_load(x.as_ptr().add(i).cast()));}
 !v128_any_true(v128_and(low,i32x4_splat(65535)))&&x[end..].iter().all(|v|v.to_bits()&65535==0)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn classify64(x:&[f32])->Result<bool,String>{
 use core::arch::wasm32::*;
 let(mut low,mut peak)=(i32x4_splat(0),i32x4_splat(0));let mask=i32x4_splat(0x7fffffff);let mut i=0;
 while i+64<=x.len(){for j in(0..64).step_by(4){let bits=v128_load(x.as_ptr().add(i+j).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));}i+=64;}
 while i+4<=x.len(){let bits=v128_load(x.as_ptr().add(i).cast());low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));i+=4;}
 for v in &x[i..]{let bits=i32x4_splat(v.to_bits()as i32);low=v128_or(low,bits);peak=u32x4_max(peak,v128_and(bits,mask));}
 finish(low,peak)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn all64(x:&[f32])->bool{
 use core::arch::wasm32::*;
 let mut low=i32x4_splat(0);let mut i=0;
 while i+64<=x.len(){for j in(0..64).step_by(4){low=v128_or(low,v128_load(x.as_ptr().add(i+j).cast()));}i+=64;}
 while i+4<=x.len(){low=v128_or(low,v128_load(x.as_ptr().add(i).cast()));i+=4;}
 !v128_any_true(v128_and(low,i32x4_splat(65535)))&&x[i..].iter().all(|v|v.to_bits()&65535==0)
}
