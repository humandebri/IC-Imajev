//! Lossless fusion of the existing conv, Q/K norm, recurrence and gated norm.
use crate::{bf, delta, execute, load_prepared_weight as load_weight, Manifest, Request, Result, MAX_FLOATS};
pub fn evaluate<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: crate::prepared_weights::WeightBuffer,
{
    evaluate_bounded(r,x,m,read,MAX_FLOATS)
}
// Only the projected stage constructs this internal buffer. Wire bounds stay
// unchanged; n<=132 and h<=16 bound this fused scratch to 1,089,664 floats.
#[cfg(feature="experimental-delta-projected")]
pub(super) fn evaluate_projected<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:crate::prepared_weights::WeightBuffer {
    if r.dims.first().is_none_or(|n|*n>132) {return Err("projected Delta scratch bound".into());}
    evaluate_bounded(r,x,m,read,1_089_664)
}
fn evaluate_bounded<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F,limit:usize)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:crate::prepared_weights::WeightBuffer {
    if r.dims.len() != 4
        || !crate::lossless_encoding(&r.encoding)
        || r.aux.len() != 1
        || !r.scalars.is_empty()
        || !r.tensor.ends_with(".linear_attn.conv1d.weight")
        || r.aux[0] != r.tensor.replace(".conv1d.weight", ".norm.weight")
    {
        return Err("delta stage metadata".into());
    }
    let (n, h, first, keep) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3]);
    if n == 0
        || n > 512
        || h == 0
        || h > 16
        || h % 2 != 0
        || first > 30
        || first % 2 != 0
        || first + h > 32
        || keep > 1
    {
        return Err("delta stage shape".into());
    }
    let channels = h * 256;
    let window = (n + 3) * channels;
    let count = n * h * 128;
    let expected = window + count + 2 * n * h + h * 16384;
    if x.len() != expected || expected > limit {
        return Err("delta stage input".into());
    }
    let conv = m
        .tensors
        .iter()
        .find(|t| t.name == r.tensor)
        .ok_or("delta stage conv")?;
    let norm = m
        .tensors
        .iter()
        .find(|t| t.name == r.aux[0])
        .ok_or("delta stage norm")?;
    if conv.rows != 8192 || conv.cols != 4 || norm.rows.checked_mul(norm.cols) != Some(128) {
        return Err("delta stage weights".into());
    }
    let mut weights = Vec::with_capacity(channels * 4);
    let mut bytes = 0;
    let mut cr = r.clone();
    cr.op = "conv_state".into();
    cr.aux.clear();
    for (start, rows) in [
        (first / 2 * 128, h / 2 * 128),
        (2048 + first / 2 * 128, h / 2 * 128),
        (4096 + first * 128, h * 128),
    ] {
        cr.dims = vec![n, rows, 4, start];
        let (w, b) = load_weight(conv, &cr, &mut *read)?;
        weights.extend_from_slice(&w);
        bytes += b;
    }
    cr.dims = vec![n, channels, 4, 0];
    let convolved = execute(&cr, &x[..window], &weights)?;
    let (norm_weight, b) = load_weight(norm, r, &mut *read)?;
    bytes += b;
    let z = &x[window..window + count];
    let g = &x[window + count..window + count + n * h];
    let beta = &x[window + count + n * h..window + count + 2 * n * h];
    let states = &x[window + count + 2 * n * h..];
    let mut result = vec![0.; count];
    let mut final_states = Vec::new();
    let mut q = Vec::new();
    let mut k = Vec::new();
    for head in 0..h {
        let gather = |start: usize| -> Vec<f32> {
            (0..n)
                .flat_map(|t| {
                    convolved[t * channels + start..t * channels + start + 128]
                        .iter()
                        .copied()
                })
                .collect()
        };
        let mut nr = r.clone();
        nr.op = "rms_scaled".into();
        nr.dims = vec![n, 128];
        nr.aux.clear();
        nr.scalars = vec![1e-6, 1. / 128.];
        // Each adjacent value-head pair uses the identical Q/K head. Retain the
        // exact rounded normalization result for the second member of the pair.
        if head % 2 == 0 {
            q = execute(&nr, &gather(head / 2 * 128), &[])?;
            nr.scalars[1] = 128f32.sqrt().recip();
            k = execute(&nr, &gather(h / 2 * 128 + head / 2 * 128), &[])?;
        }
        let v = gather(h * 128 + head * 128);
        let gh: Vec<_> = (0..n).map(|t| g[t * h + head]).collect();
        let bh: Vec<_> = (0..n).map(|t| beta[t * h + head]).collect();
        let mut state = states[head * 16384..(head + 1) * 16384].to_vec();
        let mut values: Vec<_> = delta(&q, &k, &v, &gh, &bh, &mut state, 128, 128)?
            .into_iter()
            .map(bf)
            .collect();
        for t in 0..n {
            values.extend_from_slice(&z[(t * h + head) * 128..(t * h + head + 1) * 128]);
        }
        nr.op = "gated_norm".into();
        nr.scalars = vec![1e-6];
        let gated = execute(&nr, &values, &norm_weight)?;
        for t in 0..n {
            result[(t * h + head) * 128..(t * h + head + 1) * 128]
                .copy_from_slice(&gated[t * 128..(t + 1) * 128]);
        }
        if keep == 1 {
            final_states.extend(state);
        }
    }
    result.extend(final_states);
    if !result.iter().all(|v| v.is_finite()) {
        return Err("delta stage output".into());
    }
    Ok((result, bytes))
}

/// Preserve the two integer projections and BF16 gate boundaries in one request.
pub fn gates<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: crate::prepared_weights::WeightBuffer,
{
    gates_cached(r, x, m, read, None)
}
pub(super) fn gates_cached<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F,
    cached: Option<&crate::int8_kernel::QuantizedRows>) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<B>,
    B: crate::prepared_weights::WeightBuffer,
{
    if r.dims.len() != 1
        || r.dims[0] == 0
        || r.dims[0] > 256
        || !crate::lossless_encoding(&r.encoding)
        || !r.aux.is_empty()
        || !r.scalars.is_empty()
        || !r.tensor.ends_with(".linear_attn.in_proj_a.weight")
    {
        return Err("fused gate metadata".into());
    }
    let name = r
        .tensor
        .strip_suffix(".in_proj_a.weight")
        .ok_or("gate prefix")?;
    // This variant intentionally handles the fixed adapter's non-LoRA A/B projections only.
    if m.tensors.iter().any(|t| {
        t.name == format!("{name}.in_proj_a.lora_A.weight")
            || t.name == format!("{name}.in_proj_b.lora_A.weight")
    }) {
        return Err("gate adapter unsupported".into());
    }
    let mut pr = r.clone();
    pr.op = "linear_integer_bf16".into();
    pr.dims = vec![r.dims[0], 32, 2560, 0];
    let owned;
    let q = if let Some(q) = cached { q } else {
        owned = crate::profile::measure("activation_quantize", ||
            crate::int8_kernel::quantize_rows(x, r.dims[0], 2560))?;
        &owned
    };
    let (mut a, mut bytes) = crate::evaluate_integer(&pr, x, m, &mut *read, Some(q))?;
    pr.tensor = format!("{name}.in_proj_b.weight");
    let (b, used) = crate::evaluate_integer(&pr, x, m, &mut *read, Some(q))?;
    a.extend(b);
    bytes += used;
    pr.op = "delta_gates".into();
    pr.dims = vec![r.dims[0], 32];
    pr.tensor = format!("{name}.A_log");
    pr.aux = vec![format!("{name}.dt_bias")];
    let mut weights = Vec::new();
    for name in [&pr.tensor, &pr.aux[0]] {
        let tensor = m
            .tensors
            .iter()
            .find(|t| &t.name == name)
            .ok_or("gate tensor")?;
        let (w, used) = load_weight(tensor, &pr, &mut *read)?;
        weights.extend_from_slice(&w);
        bytes += used;
    }
    Ok((execute(&pr, &a, &weights)?, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(op: &str, dims: Vec<usize>) -> Request {
        Request {
            version: 1,
            model: "a".repeat(64),
            pack_hash: "b".repeat(64),
            input_hash: "c".repeat(64),
            step: 0,
            op: op.into(),
            tensor: "model.language_model.layers.0.linear_attn.conv1d.weight".into(),
            dims,
            scalars: vec![],
            aux: vec!["model.language_model.layers.0.linear_attn.norm.weight".into()],
            encoding: "bf16-exact".into(),
        }
    }
    #[test]
    fn invalid_stage_bounds_are_rejected_before_weight_reads() {
        let m = Manifest {
            version: 1,
            model: "a".repeat(64),
            pack_hash: "b".repeat(64),
            bytes: 0,
            tensors: vec![],
        };
        for dims in [
            vec![],
            vec![0, 2, 0, 0],
            vec![513, 2, 0, 0],
            vec![1, 1, 0, 0],
            vec![1, 18, 0, 0],
            vec![1, 2, 31, 0],
            vec![1, 4, 30, 0],
            vec![1, 2, 0, 2],
            vec![usize::MAX, 2, 0, 0],
            vec![1, 2, usize::MAX, 0],
        ] {
            let r = request("delta_stage_bf16", dims);
            assert!(evaluate(&r, &[], &m, &mut |_, _| -> Result<Vec<u8>> { panic!(
                "invalid request read weights"
            ) })
            .is_err());
        }
    }
    #[test]
    fn fused_gate_rejects_adapter_before_evaluation() {
        let mut r = request("delta_gates_integer", vec![1]);
        r.tensor = r.tensor.replace("conv1d", "in_proj_a");
        r.aux.clear();
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![crate::Tensor {
                name: r.tensor.replace(".weight", ".lora_A.weight"),
                offset: 0,
                rows: 64,
                cols: 2560,
                dtype: "f32".into(),
                bytes: 0,
            }],
        };
        assert_eq!(
            gates(&r, &[], &m, &mut |_, _| -> Result<Vec<u8>> { panic!(
                "unsupported adapter read weights"
            ) })
            .unwrap_err(),
            "gate adapter unsupported"
        );
    }
}

/// Whole QKV plus the small A/B gate projections for bounded short inputs.
pub fn input<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: crate::prepared_weights::WeightBuffer,
{
    if r.dims.len() != 1
        || r.dims[0] == 0
        || r.dims[0] > 96
        || x.len() != r.dims[0] * 2560
        || !crate::lossless_encoding(&r.encoding)
        || !r.aux.is_empty()
        || !r.scalars.is_empty()
        || !r.tensor.ends_with(".linear_attn.in_proj_qkv.weight")
    {
        return Err("delta input metadata".into());
    }
    let prefix = r.tensor.strip_suffix(".weight").ok_or("QKV prefix")?;
    let mut pr = r.clone();
    pr.op = "lora_integer".into();
    pr.dims = vec![r.dims[0], 8192, 2560, 0];
    pr.scalars = vec![2.];
    pr.aux = vec![
        format!("{prefix}.lora_A.weight"),
        format!("{prefix}.lora_B.weight"),
    ];
    let q = crate::profile::measure("activation_quantize", ||
        crate::int8_kernel::quantize_rows(x, r.dims[0], 2560))?;
    let (mut mixed, bytes) = crate::evaluate_integer(&pr, x, m, &mut *read, Some(&q))?;
    let mut gr = r.clone();
    gr.op = "delta_gates_integer".into();
    gr.tensor = r.tensor.replace(".in_proj_qkv.weight", ".in_proj_a.weight");
    let (gate, gate_bytes) = gates_cached(&gr, x, m, &mut *read, Some(&q))?;
    mixed.extend(gate);
    Ok((mixed, bytes + gate_bytes))
}
