//! Append exact little-endian F32 bits with one reserve/copy on Wasm.
pub fn append(p:&mut Vec<u8>,x:&[f32]){
 #[cfg(target_arch="wasm32")]
 {
  let bytes=x.len().checked_mul(core::mem::size_of::<f32>()).expect("F32 byte length");
  // SAFETY: F32 has no padding and all its initialized representation bytes
  // are readable as U8. Wasm memory is little endian. Safe callers give an
  // independent F32 slice; Vec's mutable borrow prevents overlapping inputs.
  let raw=unsafe{core::slice::from_raw_parts(x.as_ptr().cast::<u8>(),bytes)};
  p.extend_from_slice(raw);
 }
 #[cfg(not(target_arch="wasm32"))]
 for v in x{p.extend_from_slice(&v.to_le_bytes());}
}
