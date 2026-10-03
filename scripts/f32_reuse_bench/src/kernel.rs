//! Fail-closed nine-I32 ABI for separately validated WAT body replacement.
#[export_name="__imajev_pair_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn accumulate(q:*const f32,w:*const f32,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize){let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;for i in 0..n*32{core::ptr::write_volatile(out.add(i),f32::from_bits(marker|0x7fc00000));}}
