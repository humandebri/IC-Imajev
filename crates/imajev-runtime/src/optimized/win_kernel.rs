//! Body replaced by a type-checked generated Wasm function after linking.
//! The caller establishes every fixed K256 span and initialized token row.
pub(crate) const W:usize=128;
#[export_name="__imajev_win7_wide_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*W{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32770)|0x7fc00000));}
}

#[export_name="__imajev_win7_168_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide168(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*168{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32771)|0x7fc00000));}
}

#[export_name="__imajev_win7_32_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide32(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*32{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32772)|0x7fc00000));}
}

#[export_name="__imajev_win7_wide_accumulate_seed"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide_seed(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*W{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32773)|0x7fc00000));}
}
#[export_name="__imajev_win7_168_accumulate_seed"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide168_seed(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*168{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32774)|0x7fc00000));}
}
#[export_name="__imajev_win7_32_accumulate_seed"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide32_seed(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*32{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32775)|0x7fc00000));}
}
#[export_name="__imajev_win7_160_accumulate"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide160(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*160{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32776)|0x7fc00000));}
}
#[export_name="__imajev_win7_160_accumulate_seed"]
#[inline(never)]
pub(crate) unsafe extern "C" fn wide160_seed(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;
 for i in 0..n*160{core::ptr::write_volatile(sums.add(i),f32::from_bits((marker^32777)|0x7fc00000));}
}