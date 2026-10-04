//! RNE block256 quantization, with the same divide/nearest/clamp order as scalar.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn block(x: *const f32, q: *mut i16) -> f32 {
    use core::arch::wasm32::*;
    let mask = i32x4_splat(0x7fffffff);
    let mut peak = f32x4_splat(0.);
    for i in (0..256).step_by(4) {
        peak = f32x4_max(peak, v128_and(v128_load(x.add(i).cast()), mask));
    }
    peak = f32x4_max(peak, i32x4_shuffle::<2, 3, 0, 1>(peak, peak));
    peak = f32x4_max(peak, i32x4_shuffle::<1, 0, 3, 2>(peak, peak));
    let max = f32x4_extract_lane::<0>(peak);
    let scale = if max == 0. {
        1.
    } else {
        (max / 127.).max(f32::from_bits(1))
    };
    let denominator = f32x4_splat(scale);
    let low = f32x4_splat(-127.);
    let high = f32x4_splat(127.);
    for i in (0..256).step_by(8) {
        let a = f32x4_min(
            high,
            f32x4_max(
                low,
                f32x4_nearest(f32x4_div(v128_load(x.add(i).cast()), denominator)),
            ),
        );
        let b = f32x4_min(
            high,
            f32x4_max(
                low,
                f32x4_nearest(f32x4_div(v128_load(x.add(i + 4).cast()), denominator)),
            ),
        );
        v128_store(
            q.add(i).cast(),
            i16x8_narrow_i32x4(i32x4_trunc_sat_f32x4(a), i32x4_trunc_sat_f32x4(b)),
        );
    }
    scale
}
