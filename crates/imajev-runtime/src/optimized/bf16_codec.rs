//! Compatibility path; implementation lives in inference-core.
pub use inference_core::bf16::*;

pub fn all_bf16(x:&[f32])->bool {
 #[cfg(target_arch="wasm32")]unsafe{return crate::bf16_predicates::all64(x);}
 #[cfg(not(target_arch="wasm32"))]inference_core::bf16::all_bf16(x)
}
pub fn classify_finite(x:&[f32])->crate::Result<bool> {
 #[cfg(target_arch="wasm32")]unsafe{return crate::bf16_predicates::classify64(x);}
 #[cfg(not(target_arch="wasm32"))]inference_core::bf16::classify_finite(x)
}
