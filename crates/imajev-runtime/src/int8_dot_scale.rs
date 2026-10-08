//! Exact block256 integer projection with ordered F32 scaling.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn accumulate<const R: usize, const C: usize>(
    q: *const i16, w: *const i8, cols: usize, start: usize,
    sx: *const f32, stride: usize, sw: *const f32, sums: &mut [[f32; C]; R],
) {
    crate::int8_tile::accumulate::<R, C>(q, w, cols, start, sx, stride, sw, sums);
}
