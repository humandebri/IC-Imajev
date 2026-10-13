#![cfg(feature = "experimental-single-quad")]

use imajev_runtime::{int8_kernel, output_pairs::{self, PreparedPairs}};

#[test]
fn single_token_quadrants_keep_block_order_and_shifted_view_bounds() {
    for cols in [256, 512, 2560, 9216] {
        let rows = 512;
        let raw: Vec<i8> = (0..rows * cols)
            .map(|i| [-128, 127, 0, -1, 1, -93, 41][(i + i / cols) % 7]).collect();
        let scales: Vec<f32> = (0..rows).map(|i| (i + 1) as f32 * 0.00123).collect();
        let mut bytes: Vec<u8> = raw.iter().map(|v| *v as u8).collect();
        for scale in &scales { bytes.extend(scale.to_le_bytes()); }
        let fixed = PreparedPairs::from_le_bytes(&bytes, rows, cols).unwrap();
        for zero in [false, true] {
            let x: Vec<f32> = (0..cols).map(|i| if zero { -0. } else {
                [-127., 127., -0., 0., 0.00123, -41.][i % 6] * (1 + i / 256 % 5) as f32
            }).collect();
            let q = int8_kernel::quantize_rows(&x, 1, cols).unwrap();
            // Aligned dispatch, an unaligned fallback, and the final row boundary.
            for (start, count) in [(0, 512), (4, 32), (2, 8), (504, 8)] {
                let view = fixed.original_view(start * cols, count * cols).unwrap();
                let actual = output_pairs::project(&q, &view.packed().unwrap(), &scales[start..start + count]).unwrap();
                let expected = int8_kernel::project(&q, &raw[start * cols..(start + count) * cols], &scales[start..start + count], count).unwrap();
                assert!(actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "cols={cols}, zero={zero}, start={start}, count={count}");
                assert!(!view.original_was_decoded());
            }
        }
    }
}
