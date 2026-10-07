//! Exact little endian byte decoding; trailing incomplete F32 bytes ignored.
#[inline]
pub fn decode(bytes:&[u8])->Vec<f32>{
 #[cfg(target_arch="wasm32")]{
  let len=bytes.len()/4;
  let mut result=Vec::<f32>::with_capacity(len);
  // SAFETY: all F32 bit representations are valid. Destination allocation has
  // space for len initialized values after the copy. Raw pointers avoid
  // forming references to uninitialized F32. Wasm is little endian, and the
  // source bytes and fresh allocation cannot overlap; byte source may be unaligned.
  unsafe{core::ptr::copy_nonoverlapping(bytes.as_ptr(),result.as_mut_ptr().cast::<u8>(),len*4);result.set_len(len);}
  result
 }
 #[cfg(not(target_arch="wasm32"))]{bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()}
}
