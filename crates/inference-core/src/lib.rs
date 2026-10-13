//! Model-independent primitives extracted from the Imajev inference runtime.
//! SIMD lanes represent independent tokens; each dot retains column order.
//! No model manifest, tensor names, Candid, storage or network dependencies.
pub type Result<T> = std::result::Result<T, String>;
/// Existing execution bound retained during extraction; not a model dimension.
pub const MAX_FLOATS: usize = 900_000;
pub mod bf16;
pub mod block256;
pub mod linear;
