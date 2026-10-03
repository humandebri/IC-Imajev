// Generated complete writes, before Vec length is committed.
use core::arch::wasm32::*;
#[target_feature(enable="simd128")]
pub(super) unsafe fn weights(w:&[i8],rows:usize,cols:usize,out:*mut i16) {
 let stride=cols/4;let groups=rows/4;
 for group in 0..groups {for block in 0..cols/256 {
  let dst:[*mut i16;49]=core::array::from_fn(|m|out.add(m*groups*stride+group*stride+block*64));
  for k in (0..64).step_by(8) {
   let x0=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+0)*cols+block*256+0+k).cast()));
   let x1=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+1)*cols+block*256+0+k).cast()));
   let x2=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+2)*cols+block*256+0+k).cast()));
   let x3=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+3)*cols+block*256+0+k).cast()));
   let x4=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+0)*cols+block*256+64+k).cast()));
   let x5=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+1)*cols+block*256+64+k).cast()));
   let x6=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+2)*cols+block*256+64+k).cast()));
   let x7=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+3)*cols+block*256+64+k).cast()));
   let x8=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+0)*cols+block*256+128+k).cast()));
   let x9=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+1)*cols+block*256+128+k).cast()));
   let x10=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+2)*cols+block*256+128+k).cast()));
   let x11=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+3)*cols+block*256+128+k).cast()));
   let x12=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+0)*cols+block*256+192+k).cast()));
   let x13=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+1)*cols+block*256+192+k).cast()));
   let x14=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+2)*cols+block*256+192+k).cast()));
   let x15=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add((group*4+3)*cols+block*256+192+k).cast()));
   v128_store(dst[0].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x5),x10),x15));
   v128_store(dst[1].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x10));
   v128_store(dst[2].add(k).cast(),i16x8_sub(i16x8_add(i16x8_sub(i16x8_add(i16x8_splat(0),x1),x5),x11),x15));
   v128_store(dst[3].add(k).cast(),i16x8_add(i16x8_sub(i16x8_add(i16x8_sub(i16x8_splat(0),x0),x4),x10),x14));
   v128_store(dst[4].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x5),x15));
   v128_store(dst[5].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x1),x10),x11));
   v128_store(dst[6].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x4),x5),x14),x15));
   v128_store(dst[7].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x5));
   v128_store(dst[8].add(k).cast(),i16x8_add(i16x8_splat(0),x0));
   v128_store(dst[9].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x1),x5));
   v128_store(dst[10].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x0),x4));
   v128_store(dst[11].add(k).cast(),i16x8_add(i16x8_splat(0),x5));
   v128_store(dst[12].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x1));
   v128_store(dst[13].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x4),x5));
   v128_store(dst[14].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x2),x7),x10),x15));
   v128_store(dst[15].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x2),x10));
   v128_store(dst[16].add(k).cast(),i16x8_add(i16x8_sub(i16x8_sub(i16x8_add(i16x8_splat(0),x3),x7),x11),x15));
   v128_store(dst[17].add(k).cast(),i16x8_sub(i16x8_add(i16x8_add(i16x8_sub(i16x8_splat(0),x2),x6),x10),x14));
   v128_store(dst[18].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x7),x15));
   v128_store(dst[19].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x2),x3),x10),x11));
   v128_store(dst[20].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x6),x7),x14),x15));
   v128_store(dst[21].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x0),x5),x8),x13));
   v128_store(dst[22].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x0),x8));
   v128_store(dst[23].add(k).cast(),i16x8_sub(i16x8_add(i16x8_add(i16x8_sub(i16x8_splat(0),x1),x5),x9),x13));
   v128_store(dst[24].add(k).cast(),i16x8_add(i16x8_sub(i16x8_sub(i16x8_add(i16x8_splat(0),x0),x4),x8),x12));
   v128_store(dst[25].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x5),x13));
   v128_store(dst[26].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x0),x1),x8),x9));
   v128_store(dst[27].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x4),x5),x12),x13));
   v128_store(dst[28].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x10),x15));
   v128_store(dst[29].add(k).cast(),i16x8_add(i16x8_splat(0),x10));
   v128_store(dst[30].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x11),x15));
   v128_store(dst[31].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x10),x14));
   v128_store(dst[32].add(k).cast(),i16x8_add(i16x8_splat(0),x15));
   v128_store(dst[33].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x10),x11));
   v128_store(dst[34].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x14),x15));
   v128_store(dst[35].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x2),x5),x7));
   v128_store(dst[36].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x2));
   v128_store(dst[37].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x1),x3),x5),x7));
   v128_store(dst[38].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x0),x2),x4),x6));
   v128_store(dst[39].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x5),x7));
   v128_store(dst[40].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x1),x2),x3));
   v128_store(dst[41].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x4),x5),x6),x7));
   v128_store(dst[42].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x8),x10),x13),x15));
   v128_store(dst[43].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x8),x10));
   v128_store(dst[44].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x9),x11),x13),x15));
   v128_store(dst[45].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x8),x10),x12),x14));
   v128_store(dst[46].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x13),x15));
   v128_store(dst[47].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x8),x9),x10),x11));
   v128_store(dst[48].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x12),x13),x14),x15));
  }
 }}
}
#[target_feature(enable="simd128")]
pub(super) unsafe fn inputs(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,out:*mut i16) {
 let cols=q.cols();let stride=cols/4;
 for group in 0..groups {for block in 0..cols/256 {
  let dst:[*mut i16;49]=core::array::from_fn(|m|out.add(m*groups*stride+group*stride+block*64));
  for k in (0..64).step_by(8) {
   let x0=v128_load(q.values().as_ptr().add((group*4+0)*cols+block*256+0+k).cast());
   let x1=v128_load(q.values().as_ptr().add((group*4+0)*cols+block*256+64+k).cast());
   let x2=v128_load(q.values().as_ptr().add((group*4+0)*cols+block*256+128+k).cast());
   let x3=v128_load(q.values().as_ptr().add((group*4+0)*cols+block*256+192+k).cast());
   let x4=v128_load(q.values().as_ptr().add((group*4+1)*cols+block*256+0+k).cast());
   let x5=v128_load(q.values().as_ptr().add((group*4+1)*cols+block*256+64+k).cast());
   let x6=v128_load(q.values().as_ptr().add((group*4+1)*cols+block*256+128+k).cast());
   let x7=v128_load(q.values().as_ptr().add((group*4+1)*cols+block*256+192+k).cast());
   let x8=v128_load(q.values().as_ptr().add((group*4+2)*cols+block*256+0+k).cast());
   let x9=v128_load(q.values().as_ptr().add((group*4+2)*cols+block*256+64+k).cast());
   let x10=v128_load(q.values().as_ptr().add((group*4+2)*cols+block*256+128+k).cast());
   let x11=v128_load(q.values().as_ptr().add((group*4+2)*cols+block*256+192+k).cast());
   let x12=v128_load(q.values().as_ptr().add((group*4+3)*cols+block*256+0+k).cast());
   let x13=v128_load(q.values().as_ptr().add((group*4+3)*cols+block*256+64+k).cast());
   let x14=v128_load(q.values().as_ptr().add((group*4+3)*cols+block*256+128+k).cast());
   let x15=v128_load(q.values().as_ptr().add((group*4+3)*cols+block*256+192+k).cast());
   v128_store(dst[0].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x5),x10),x15));
   v128_store(dst[1].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x4),x5),x14),x15));
   v128_store(dst[2].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x10));
   v128_store(dst[3].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x5),x15));
   v128_store(dst[4].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x1),x10),x11));
   v128_store(dst[5].add(k).cast(),i16x8_add(i16x8_sub(i16x8_add(i16x8_sub(i16x8_splat(0),x0),x4),x10),x14));
   v128_store(dst[6].add(k).cast(),i16x8_sub(i16x8_add(i16x8_sub(i16x8_add(i16x8_splat(0),x1),x5),x11),x15));
   v128_store(dst[7].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x8),x10),x13),x15));
   v128_store(dst[8].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x12),x13),x14),x15));
   v128_store(dst[9].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x8),x10));
   v128_store(dst[10].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x13),x15));
   v128_store(dst[11].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x8),x9),x10),x11));
   v128_store(dst[12].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x8),x10),x12),x14));
   v128_store(dst[13].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x9),x11),x13),x15));
   v128_store(dst[14].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x5));
   v128_store(dst[15].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x4),x5));
   v128_store(dst[16].add(k).cast(),i16x8_add(i16x8_splat(0),x0));
   v128_store(dst[17].add(k).cast(),i16x8_add(i16x8_splat(0),x5));
   v128_store(dst[18].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x1));
   v128_store(dst[19].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x0),x4));
   v128_store(dst[20].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x1),x5));
   v128_store(dst[21].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x10),x15));
   v128_store(dst[22].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x14),x15));
   v128_store(dst[23].add(k).cast(),i16x8_add(i16x8_splat(0),x10));
   v128_store(dst[24].add(k).cast(),i16x8_add(i16x8_splat(0),x15));
   v128_store(dst[25].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x10),x11));
   v128_store(dst[26].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x10),x14));
   v128_store(dst[27].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x11),x15));
   v128_store(dst[28].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x2),x5),x7));
   v128_store(dst[29].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x4),x5),x6),x7));
   v128_store(dst[30].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x0),x2));
   v128_store(dst[31].add(k).cast(),i16x8_add(i16x8_add(i16x8_splat(0),x5),x7));
   v128_store(dst[32].add(k).cast(),i16x8_add(i16x8_add(i16x8_add(i16x8_add(i16x8_splat(0),x0),x1),x2),x3));
   v128_store(dst[33].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x0),x2),x4),x6));
   v128_store(dst[34].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x1),x3),x5),x7));
   v128_store(dst[35].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x0),x5),x8),x13));
   v128_store(dst[36].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x4),x5),x12),x13));
   v128_store(dst[37].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x0),x8));
   v128_store(dst[38].add(k).cast(),i16x8_add(i16x8_sub(i16x8_splat(0),x5),x13));
   v128_store(dst[39].add(k).cast(),i16x8_add(i16x8_add(i16x8_sub(i16x8_sub(i16x8_splat(0),x0),x1),x8),x9));
   v128_store(dst[40].add(k).cast(),i16x8_add(i16x8_sub(i16x8_sub(i16x8_add(i16x8_splat(0),x0),x4),x8),x12));
   v128_store(dst[41].add(k).cast(),i16x8_sub(i16x8_add(i16x8_add(i16x8_sub(i16x8_splat(0),x1),x5),x9),x13));
   v128_store(dst[42].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x2),x7),x10),x15));
   v128_store(dst[43].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x6),x7),x14),x15));
   v128_store(dst[44].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x2),x10));
   v128_store(dst[45].add(k).cast(),i16x8_sub(i16x8_add(i16x8_splat(0),x7),x15));
   v128_store(dst[46].add(k).cast(),i16x8_sub(i16x8_sub(i16x8_add(i16x8_add(i16x8_splat(0),x2),x3),x10),x11));
   v128_store(dst[47].add(k).cast(),i16x8_sub(i16x8_add(i16x8_add(i16x8_sub(i16x8_splat(0),x2),x6),x10),x14));
   v128_store(dst[48].add(k).cast(),i16x8_add(i16x8_sub(i16x8_sub(i16x8_add(i16x8_splat(0),x3),x7),x11),x15));
  }
 }}
}
