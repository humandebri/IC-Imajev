use inference_core::{block256, linear::matrix_baseline};

fn main() -> inference_core::Result<()> {
    // Two token rows and three output rows; weights are output-major.
    let output = matrix_baseline(&[1., 2., 3., 4.], &[1., 0., 0., 1., 0.5, -0.5], 2, 3, 2)?;
    let mut payload = Vec::new();
    block256::append(&mut payload, &output)?;
    let restored = block256::decode(&payload)?;
    assert!(output
        .iter()
        .zip(&restored)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    println!("{restored:?} ({} payload bytes)", payload.len());
    Ok(())
}
