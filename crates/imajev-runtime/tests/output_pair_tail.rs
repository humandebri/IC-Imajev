#![cfg(feature = "experimental-prepared-output-pairs")]

use imajev_runtime::{int8_kernel, output_pairs::{self, PreparedPairs}};

#[test]
fn partial_output_views_preserve_every_block_scale_and_end_boundary() {
    for cols in [512, 2560] {
        let rows = 64;
        let raw: Vec<i8> = (0..rows * cols)
            .map(|i| [-128, 127, -1, 0, 1, -93, 41][i % 7])
            .collect();
        let scales: Vec<f32> = (0..rows).map(|i| (i + 1) as f32 * 0.00123).collect();
        let mut bytes: Vec<u8> = raw.iter().map(|v| *v as u8).collect();
        for scale in &scales { bytes.extend(scale.to_le_bytes()); }
        let fixed = PreparedPairs::from_le_bytes(&bytes, rows, cols).unwrap();
        for n in [1, 45, 87, 132] {
            let x: Vec<f32> = (0..n * cols)
                .map(|i| [-127., 127., -0., 0., 0.00123, -41.][i % 6]
                    * (1 + i / 256 % 5) as f32)
                .collect();
            let q = int8_kernel::quantize_rows(&x, n, cols).unwrap();
            for (start, count) in [(8, 8), (16, 16), (40, 24), (8, 40), (8, 56)] {
                let view = fixed.original_view(start * cols, count * cols).unwrap();
                let packed = view.packed().expect("eight-row query boundary");
                let actual = output_pairs::project(&q, &packed, &scales[start..start + count]).unwrap();
                let expected = int8_kernel::project(&q, &raw[start * cols..(start + count) * cols],
                    &scales[start..start + count], count).unwrap();
                assert!(!view.original_was_decoded());
                assert_eq!(actual.len(), n * count);
                assert!(actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "n={n}, cols={cols}, start={start}, rows={count}");
            }
        }
    }
}
