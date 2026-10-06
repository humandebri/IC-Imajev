//! Body replaced by a type-checked generated Wasm function after linking.
//! The caller establishes every fixed K256 span and initialized token row.
#[export_name="__imajev_pair_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn accumulate(_q:*const i16,_w:*const i8,_cols:usize,_start:usize,_sx:*const f32,_stride:usize,_sw:*const f32,_sums:*mut f32,_n:usize) {
 // Fail closed: an unpatched diagnostic must not produce valid inference.
 // The placeholder must have the same whole-buffer memory effect as the
 // replacement. Otherwise interprocedural optimization may retain zeros
 // for cells it believes the placeholder cannot write.
 // Every ABI argument must affect this opaque write. An unused stub
 // parameter can be replaced by undef/zero at the call site, even though
 // the subsequently injected body needs its real value.
 let marker=core::hint::black_box((_q as usize)^(_w as usize)^_cols^_start^(_sx as usize)^_stride^(_sw as usize)^(_sums as usize)^_n) as u32;
 for i in 0.._n*32 {core::ptr::write_volatile(_sums.add(i), f32::from_bits(marker|0x7fc00000));}
}

// Direct Cargo builds default to128. The benchmark builder explicitly selects
// a checked64/128/256 cfg; unpatched stubs remain fail-closed.
#[cfg(not(any(wide64,wide128,wide256)))] pub(crate) const W:usize=128;
#[cfg(wide64)] pub(crate) const W:usize=64;
#[cfg(wide128)] pub(crate) const W:usize=128;
#[cfg(wide256)] pub(crate) const W:usize=256;
#[export_name="__imajev_s1_wide_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*W{core::ptr::write_volatile(sums.add(i),f32::from_bits(marker|0x7fc00000));}
}
