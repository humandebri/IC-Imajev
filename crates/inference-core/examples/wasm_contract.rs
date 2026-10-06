//! Executable SIMD/codec contract check for wasm32-unknown-unknown.
use inference_core::{
    block256,
    linear::{matrix_baseline, matrix_reference},
};

#[no_mangle]
pub extern "C" fn check_contract() -> u32 {
    for n in [1, 3, 4, 7, 32, 65, 87, 132] {
        for rows in [1, 17, 64] {
            let cols = 256;
            let x: Vec<_> = (0..n * cols)
                .map(|i| [0., -0., 0.125, -3.75, f32::from_bits(1), 16.25][i % 6])
                .collect();
            let w: Vec<_> = (0..rows * cols)
                .map(|i| [-0., 0.125, -0.75, 4.5, f32::from_bits(1)][i % 5])
                .collect();
            let reference = matrix_reference(&x, &w, n, rows, cols).unwrap();
            let output = matrix_baseline(&x, &w, n, rows, cols).unwrap();
            if !reference
                .iter()
                .zip(&output)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return 1;
            }
            let mut payload = Vec::new();
            block256::append(&mut payload, &output).unwrap();
            let restored = block256::decode(&payload).unwrap();
            if !reference
                .iter()
                .zip(&restored)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return 2;
            }
        }
    }
    for bits in 0..=u16::MAX {
        let value = f32::from_bits((bits as u32) << 16);
        if inference_core::bf16::classify_finite(&[value]).is_ok() != value.is_finite() {
            return 3;
        }
    }
    0
}
