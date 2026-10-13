use inference_core::{
    block256,
    linear::{matrix_baseline, matrix_reference},
};

#[test]
fn independent_lanes_preserve_scalar_order_and_codec_bits() {
    for n in [1, 3, 4, 7, 16, 31, 32, 45, 64, 65, 87, 128, 132] {
        for rows in [1, 15, 16, 17, 64] {
            for cols in [1, 7, 64, 256] {
                let x: Vec<_> = (0..n * cols)
                    .map(|i| [0., -0., 0.125, -3.75, f32::from_bits(1), 16.25][i % 6])
                    .collect();
                let w: Vec<_> = (0..rows * cols)
                    .map(|i| [-0., 0.125, -0.75, 4.5, f32::from_bits(1)][i % 5])
                    .collect();
                let expected = matrix_reference(&x, &w, n, rows, cols).unwrap();
                let actual = matrix_baseline(&x, &w, n, rows, cols).unwrap();
                let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&actual), bits(&expected), "{n}/{rows}/{cols}");
                let mut payload = Vec::new();
                block256::append(&mut payload, &actual).unwrap();
                assert_eq!(bits(&block256::decode(&payload).unwrap()), bits(&expected));
            }
        }
    }
}

#[test]
fn public_entry_points_reject_invalid_sizes() {
    for project in [matrix_baseline, matrix_reference] {
        assert!(project(&[], &[], 0, 0, 0).is_err());
        assert!(project(&[], &[], usize::MAX, 1, 2).is_err());
        assert!(project(&[1.], &[1.], 1, 2, 1).is_err());
        // Small operands, output exceeding the retained allocation bound.
        assert!(project(&vec![0.; 1000], &vec![0.; 1000], 1000, 1000, 1).is_err());
    }
    assert!(block256::append(&mut Vec::new(), &vec![0.; 900_001]).is_err());
}
