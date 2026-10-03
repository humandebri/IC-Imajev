//! Fuse residual/norm with MLP without returning an unused normalized tensor.
use crate::{bf, evaluate_mlp, execute, load_prepared_weight as load_weight, Manifest, Request, Result, MAX_FLOATS};

pub fn evaluate<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: crate::prepared_weights::WeightBuffer,
{
    if r.op == "add_norm_chain_bf16" {
        if !crate::lossless_encoding(&r.encoding)
            || r.dims.len() != 2
            || !r.aux.is_empty()
            || r.scalars.len() != 1
        {
            return Err("residual chain metadata".into());
        }
        let (n, cols) = (r.dims[0], r.dims[1]);
        let count = n.checked_mul(cols).ok_or("residual chain size")?;
        if n == 0 || n > 512 || cols == 0 || count > MAX_FLOATS / 3 || count * 3 != x.len() {
            return Err("residual chain bounds".into());
        }
        // Preserve both BF16 additions; summing all three before rounding differs.
        let mut pair: Vec<_> = (0..count).map(|i| bf(x[i] + x[count + i])).collect();
        pair.extend_from_slice(&x[2 * count..]);
        let mut old = r.clone();
        old.op = "add_norm_bf16".into();
        let tensor = m
            .tensors
            .iter()
            .find(|t| t.name == r.tensor)
            .ok_or("residual chain norm weight")?;
        if tensor.rows.checked_mul(tensor.cols) != Some(cols) {
            return Err("residual chain norm shape".into());
        }
        let (weight, bytes) = load_weight(tensor, &old, &mut *read)?;
        return Ok((execute(&old, &pair, &weight)?, bytes));
    }
    if !crate::lossless_encoding(&r.encoding) || r.dims.len() != 4 || r.aux.len() != 2 || r.scalars.len() != 2 {
        return Err("MLP residual norm metadata".into());
    }
    let (n, rows, cols) = (r.dims[0], r.dims[1], r.dims[2]);
    let count = n.checked_mul(cols).ok_or("MLP residual norm size")?;
    if n == 0
        || n > 512
        || rows == 0
        || rows % 8 != 0
        || cols == 0
        || cols % 256 != 0
        || count > MAX_FLOATS / 2
        || count * 2 != x.len()
        || !r.scalars.iter().all(|v| v.is_finite())
        || r.scalars[1] <= 0.
    {
        return Err("MLP residual norm bounds".into());
    }
    let prefix = r
        .tensor
        .strip_suffix(".mlp.gate_proj.weight")
        .ok_or("MLP residual norm gate")?;
    if r.aux[0] != format!("{prefix}.mlp.up_proj.weight")
        || r.aux[1] != format!("{prefix}.post_attention_layernorm.weight")
    {
        return Err("MLP residual norm names".into());
    }
    let norm = m
        .tensors
        .iter()
        .find(|t| t.name == r.aux[1])
        .ok_or("MLP residual norm weight")?;
    if norm.rows.checked_mul(norm.cols) != Some(cols) {
        return Err("MLP residual norm weight shape".into());
    }
    let mut nr = r.clone();
    nr.op = "add_norm_bf16".into();
    nr.tensor = r.aux[1].clone();
    nr.dims = vec![n, cols];
    nr.aux.clear();
    nr.scalars = vec![r.scalars[1]];
    let (weight, bytes) = load_weight(norm, &nr, &mut *read)?;
    let both = execute(&nr, x, &weight)?;
    let mut mr = r.clone();
    mr.op = "mlp_gate_up_integer".into();
    mr.aux.truncate(1);
    mr.scalars.truncate(1);
    let (output, projection_bytes) = evaluate_mlp(&mr, &both[count..], m, read)?;
    Ok((output, bytes + projection_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chain_keeps_both_bf16_roundings_and_rejects_invalid_lengths() {
        let norm = crate::Tensor {
            name: "norm".into(),
            rows: 1,
            cols: 7,
            dtype: "f32".into(),
            offset: 0,
            bytes: 28,
        };
        let m = Manifest {
            version: 1,
            model: "a".repeat(64),
            pack_hash: "b".repeat(64),
            bytes: 28,
            tensors: vec![norm],
        };
        let data: Vec<_> = (0..7).flat_map(|_| 1f32.to_le_bytes()).collect();
        let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"add_norm_chain_bf16","tensor":"norm","dims":[2,7],"scalars":[1e-6],"encoding":"bf16-exact"})).unwrap();
        let mut x = vec![1.; 14];
        x.extend(vec![0.00390625; 28]);
        let got = crate::evaluate_with_reader(&r, &x, &m, |offset, len| {
            Ok(data[offset as usize..offset as usize + len].to_vec())
        })
        .unwrap()
        .0;
        assert_eq!(&got[..14], &[1.; 14]);
        assert_ne!(bf(1. + 0.00390625 + 0.00390625), 1.);
        let mut old = r.clone();
        old.op = "add_norm_bf16".into();
        let mut pair = vec![1.; 14];
        pair.extend(vec![0.00390625; 14]);
        let expected = execute(&old, &pair, &[1.; 7]).unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(crate::evaluate_with_reader(&r, &x[..41], &m, |_, _| panic!(
            "must reject before read"
        ))
        .is_err());
    }
}
