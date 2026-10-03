//! Reuse RoPE frequencies and positions without changing F32/BF16 operations.
use crate::{
    bf, execute, load_prepared_weight as load_weight, Manifest, Request, Result, MAX_FLOATS,
};

// Computed in the same Wasm module during an owner preparation update, never
// filled by a query. Unsupported theta/rotary/positions retain the old path.
#[cfg(feature = "experimental-prepared-rope")]
const FIXED_POSITIONS: usize = 512;
#[cfg(feature = "experimental-prepared-rope")]
thread_local! {
    static FIXED_ANGLES: std::cell::RefCell<Option<Box<[(f32,f32)]>>> = const { std::cell::RefCell::new(None) };
}
#[cfg(feature = "experimental-prepared-rope")]
pub fn prepare_fixed() -> usize {
    FIXED_ANGLES.with(|table| {
        let mut table = table.borrow_mut();
        if table.is_none() {
            let positions = positions(10000000., 64, FIXED_POSITIONS, 0);
            *table = Some(positions.into_boxed_slice());
        }
        table.as_ref().unwrap().len() * std::mem::size_of::<(f32, f32)>()
    })
}
#[cfg(feature = "experimental-prepared-rope")]
pub fn fixed_bytes() -> usize {
    FIXED_ANGLES.with(|table| {
        table
            .borrow()
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<(f32, f32)>())
    })
}
#[cfg(feature = "experimental-prepared-rope")]
pub fn clear_fixed() {
    FIXED_ANGLES.with(|table| *table.borrow_mut() = None);
}

pub fn transform(r: &Request, x: &[f32]) -> Result<Vec<f32>> {
    if !matches!(r.dims.len(), 4 | 5) || r.scalars.len() != 1 {
        return Err("rope metadata".into());
    }
    let (n, width, rotary, offset) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3]);
    let heads = if r.dims.len() == 5 { r.dims[4] } else { 1 };
    let theta = r.scalars[0];
    let single = r.dims.len() == 4;
    let count = n.checked_mul(width).and_then(|v| v.checked_mul(heads));
    if n == 0
        || n > (if single { 262144 } else { 512 })
        || width == 0
        || width > (if single { 262144 } else { 256 })
        || rotary == 0
        || rotary > width
        || rotary % 2 != 0
        || offset > 262144
        || heads == 0
        || heads > 16
        || count != Some(x.len())
        || x.len() > MAX_FLOATS
        || !theta.is_finite()
        || theta <= 0.
    {
        return Err("rope shape/scalar bounds".into());
    }
    let half = rotary / 2;
    #[cfg(feature = "experimental-prepared-rope")]
    if rotary == 64
        && theta.to_bits() == 10000000f32.to_bits()
        && offset
            .checked_add(n)
            .is_some_and(|end| end <= FIXED_POSITIONS)
    {
        if let Some(output) = FIXED_ANGLES.with(|table| {
            table.borrow().as_ref().map(|angles| {
                apply_positions(
                    x,
                    n,
                    width,
                    half,
                    heads,
                    &angles[offset * half..(offset + n) * half],
                )
            })
        }) {
            return Ok(output);
        }
    }
    // Same powf arguments and multiplication as the original scalar kernel.
    let positions = positions(theta, rotary, n, offset);
    Ok(apply_positions(x, n, width, half, heads, &positions))
}

// Both preparation and fallback call the same compiled calculation. Avoid
// caller-specific constant folding for the fixed theta during preparation.
#[inline(never)]
fn positions(theta: f32, rotary: usize, n: usize, offset: usize) -> Vec<(f32, f32)> {
    let frequency: Vec<_> = (0..rotary / 2)
        .map(|i| theta.powf(-((2 * i) as f32) / rotary as f32))
        .collect();
    let mut positions = Vec::with_capacity(n * (rotary / 2));
    for t in 0..n {
        for f in &frequency {
            positions.push(((t + offset) as f32 * f).sin_cos());
        }
    }
    positions
}

fn apply_positions(
    x: &[f32],
    n: usize,
    width: usize,
    half: usize,
    heads: usize,
    positions: &[(f32, f32)],
) -> Vec<f32> {
    let mut y = x.to_vec();
    for head in 0..heads {
        for t in 0..n {
            let row = (head * n + t) * width;
            for i in 0..half {
                let (sin, cos) = positions[t * half + i];
                let (a, b) = (x[row + i], x[row + half + i]);
                y[row + i] = bf(a * cos - b * sin);
                y[row + half + i] = bf(b * cos + a * sin);
            }
        }
    }
    y
}

pub fn evaluate<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: crate::prepared_weights::WeightBuffer,
{
    if !crate::lossless_encoding(&r.encoding)
        || r.dims.len() != 5
        || r.scalars.len() != 2
        || !r.aux.is_empty()
        || !r.scalars[0].is_finite()
        || r.scalars[0] <= 0.
    {
        return Err("norm rope metadata".into());
    }
    let mut rotation = r.clone();
    rotation.op = "rope_heads".into();
    rotation.scalars = vec![r.scalars[1]];
    // Validate all bounds before reading any weight. The transform is only
    // executed after normalization; no question-dependent work moves to host.
    let (n, width, rotary, offset, heads) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3], r.dims[4]);
    if n == 0
        || n > 512
        || width == 0
        || width > 256
        || rotary == 0
        || rotary > width
        || rotary % 2 != 0
        || offset > 262144
        || heads == 0
        || heads > 16
        || n * width * heads != x.len()
        || x.len() > MAX_FLOATS
        || !r.scalars[1].is_finite()
        || r.scalars[1] <= 0.
    {
        return Err("norm rope bounds".into());
    }
    let tensor = m
        .tensors
        .iter()
        .find(|t| t.name == r.tensor)
        .ok_or("rope norm weight")?;
    if tensor.rows.checked_mul(tensor.cols) != Some(width) {
        return Err("rope norm weight shape".into());
    }
    let mut nr = r.clone();
    nr.op = "rms_bf16".into();
    nr.dims = vec![n * heads, width];
    nr.scalars = vec![r.scalars[0]];
    let (weight, bytes) = load_weight(tensor, &nr, &mut *read)?;
    let normalized = execute(&nr, x, &weight)?;
    let output = transform(&rotation, &normalized)?;
    if !output.iter().all(|v| v.is_finite()) {
        return Err("nonfinite norm rope output".into());
    }
    Ok((output, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(n: usize, h: usize, offset: usize) -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"rope_heads","tensor":"","dims":[n,16,8,offset,h],"scalars":[10000000.],"encoding":"bf16-exact"})).unwrap()
    }
    #[test]
    fn reused_angles_match_original_scalar_order_and_preserve_tail() {
        for n in [1, 45, 87, 132] {
            for h in [1, 4, 16] {
                for offset in [0, 45, 131] {
                    let r = request(n, h, offset);
                    let x: Vec<_> = (0..n * h * 16)
                        .map(|i| bf((i as f32 * 0.013).sin()))
                        .collect();
                    let mut old = x.clone();
                    for head in 0..h {
                        for t in 0..n {
                            for i in 0..4 {
                                let angle =
                                    (t + offset) as f32 * 10000000f32.powf(-((2 * i) as f32) / 8.);
                                let (s, c) = angle.sin_cos();
                                let row = (head * n + t) * 16;
                                old[row + i] = bf(x[row + i] * c - x[row + 4 + i] * s);
                                old[row + 4 + i] = bf(x[row + 4 + i] * c + x[row + i] * s);
                            }
                        }
                    }
                    assert_eq!(
                        transform(&r, &x)
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        old.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[cfg(feature = "experimental-prepared-rope")]
    #[test]
    fn fixed_table_is_exact_idempotent_and_bounded() {
        clear_fixed();
        assert_eq!(fixed_bytes(), 0);
        for (n, offset, rotary, theta) in [
            (87, 45, 64, 10000000.),
            (132, 0, 64, 10000000.),
            (1, 511, 64, 10000000.),
            (2, 511, 64, 10000000.),
            (7, 45, 8, 10000000.),
            (45, 0, 64, 500000.),
        ] {
            clear_fixed();
            let mut r = request(n, 4, offset);
            r.dims[1] = 256;
            r.dims[2] = rotary;
            r.scalars[0] = theta;
            let x: Vec<_> = (0..n * 4 * 256)
                .map(|i| bf((i as f32 * 0.013).sin()))
                .collect();
            let old = transform(&r, &x).unwrap();
            assert_eq!(prepare_fixed(), 131072);
            assert_eq!(prepare_fixed(), 131072);
            assert_eq!(fixed_bytes(), 131072);
            let new = transform(&r, &x).unwrap();
            assert_eq!(
                old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                new.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        clear_fixed();
        assert_eq!(fixed_bytes(), 0);
    }
    #[test]
    fn invalid_fusion_is_rejected_before_read() {
        let mut r = request(1, 16, 0);
        r.op = "norm_rope_heads_bf16".into();
        r.scalars = vec![1e-6, 10000000.];
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![],
        };
        r.dims[4] = usize::MAX;
        assert!(evaluate(&r, &[], &m, &mut |_, _| -> Result<Vec<u8>> {
            panic!("invalid request read weights")
        })
        .is_err());
    }
    #[test]
    fn fusion_normalizes_rows_and_rejects_nonfinite_rotation() {
        let mut r = request(1, 4, 0);
        r.op = "norm_rope_heads_bf16".into();
        r.tensor = "norm".into();
        r.scalars = vec![1e-6, 10000000.];
        let weights: Vec<_> = (0..16).map(|i| 0.75 + i as f32 / 32.).collect();
        let data: Vec<_> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: data.len() as u64,
            tensors: vec![crate::Tensor {
                name: r.tensor.clone(),
                rows: 1,
                cols: 16,
                dtype: "f32".into(),
                offset: 0,
                bytes: data.len() as u64,
            }],
        };
        let x: Vec<_> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut nr = r.clone();
        nr.op = "rms_bf16".into();
        nr.dims = vec![4, 16];
        nr.scalars = vec![1e-6];
        let expected = execute(&nr, &x, &weights).unwrap();
        let got = evaluate(&r, &x, &m, &mut |offset, len| {
            Ok(data[offset as usize..offset as usize + len].to_vec())
        })
        .unwrap()
        .0;
        // Position zero is identity rotation; each row must retain its own norm.
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        r.dims[2] = 16;
        r.dims[3] = 1;
        r.scalars[1] = f32::from_bits(1);
        assert!(evaluate(&r, &x, &m, &mut |offset, len| Ok(data
            [offset as usize..offset as usize + len]
            .to_vec()))
        .is_err());
    }
}
