//! Internal head groups with one immutable input preparation and exact prefix replay.
use crate::{
    int8_kernel::QuantizedRows, prepared_weights::WeightBuffer, Manifest, Request, Result,
};
const C: usize = 2560;
const R: usize = 64;
pub(crate) struct Preparation {
    pub(crate) q: QuantizedRows,
    pub(crate) qa: Vec<f32>,
    pub(crate) za: Vec<f32>,
    pub(crate) gates: Vec<f32>,
}
impl Preparation {
    pub(crate) fn capture<F, B>(
        r: &Request,
        norm: &[f32],
        m: &Manifest,
        read: &mut F,
    ) -> Result<(Self, u64)>
    where
        F: FnMut(u64, usize) -> Result<B>,
        B: WeightBuffer,
    {
        let n = r.dims[0];
        if !(1..=91).contains(&n) || norm.len() != n * C || !crate::bf16_codec::all_bf16(norm) {
            return Err("head preparation norm".into());
        }
        let root = r
            .tensor
            .strip_suffix(".in_proj_qkv.weight")
            .ok_or("head preparation root")?;
        let qp = crate::delta_projected::projection(r, m, r.tensor.clone(), n, 8192, 0)?;
        let zp = crate::delta_projected::projection(
            r,
            m,
            format!("{root}.in_proj_z.weight"),
            n,
            4096,
            0,
        )?;
        let q = crate::profile::measure("head_input_quantize_once", || {
            crate::int8_kernel::quantize_rows(norm, n, C)
        })?;
        let (qa, mut bytes) = crate::delta_projected::a_product(&qp, norm, m, read)?;
        let (za, used) = crate::delta_projected::a_product(&zp, norm, m, read)?;
        bytes += used;
        let mut gr = r.clone();
        gr.op = "delta_gates_integer".into();
        gr.tensor = format!("{root}.in_proj_a.weight");
        gr.dims = vec![n];
        let (gates, used) = crate::delta_stage::gates_cached(&gr, norm, m, read, Some(&q))?;
        bytes += used;
        Ok((Self { q, qa, za, gates }, bytes))
    }
    pub(crate) fn restore(n: usize, integers: &[u8], rest: &[f32]) -> Result<Self> {
        if n == 0
            || n > 91
            || rest.len() != n * (C / 256 + 2 * R + 64)
            || !rest.iter().all(|v| v.is_finite())
            || !rest[n * (C / 256 + 2 * R)..]
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        {
            return Err("head restore fields".into());
        }
        let sx = n * C / 256;
        let a = n * R;
        let q = QuantizedRows::from_bytes(n, C, integers, &rest[..sx])?;
        Ok(Self {
            q,
            qa: rest[sx..sx + a].to_vec(),
            za: rest[sx + a..sx + 2 * a].to_vec(),
            gates: rest[sx + 2 * a..].to_vec(),
        })
    }
    pub(crate) fn flat(&self, n: usize, out: &mut Vec<f32>) {
        self.q.append_wire_values(out);
        out.extend_from_slice(&self.q.scales()[..n * C / 256]);
        out.extend_from_slice(&self.qa);
        out.extend_from_slice(&self.za);
        out.extend_from_slice(&self.gates);
    }
}
pub(crate) fn group<F, B>(
    r: &Request,
    prep: &Preparation,
    history: &[f32],
    log: &[f32],
    n: usize,
    h: usize,
    first: usize,
    p: usize,
    m: &Manifest,
    read: &mut F,
) -> Result<(Vec<f32>, Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: WeightBuffer,
{
    if !(1..=91).contains(&n)
        || h == 0
        || h > 32
        || h % 2 != 0
        || first % 2 != 0
        || first.checked_add(h).is_none_or(|v| v > 32)
        || p == 0
        || p > 132
        || history.len() != 3 * h * 256
        || log.len() != p * (h / 2 * 128 + h * 128 + h)
        || !crate::bf16_codec::all_bf16(history)
        || !crate::bf16_codec::all_bf16(&log[..p * h / 2 * 128])
        || prep.q.rows() != n
        || prep.q.cols() != C
        || prep.qa.len() != n * R
        || prep.za.len() != n * R
        || prep.gates.len() != n * 64
    {
        return Err("head group bounds/precision".into());
    }
    let mut states = crate::profile::measure("head_prefix_restore_key_major", || {
        crate::delta_log::restore_key_major(p, h, log)
    })?;
    let root = r
        .tensor
        .strip_suffix(".in_proj_qkv.weight")
        .ok_or("head group tensor")?;
    let channels = h * 256;
    let mut qp = crate::delta_projected::projection(
        r,
        m,
        r.tensor.clone(),
        n,
        h / 2 * 128,
        first / 2 * 128,
    )?;
    let zp = crate::delta_projected::projection(
        r,
        m,
        format!("{root}.in_proj_z.weight"),
        n,
        h * 128,
        first * 128,
    )?;
    let mut mixed = vec![0.; n * channels];
    let mut column = 0;
    let mut bytes = 0;
    for (start, rows) in [
        (first / 2 * 128, h / 2 * 128),
        (2048 + first / 2 * 128, h / 2 * 128),
        (4096 + first * 128, h * 128),
    ] {
        qp.dims = vec![n, rows, C, start];
        let (tile, used) =
            crate::evaluate_integer_with_ax(&qp, &[], m, read, Some(&prep.q), Some(&prep.qa))?;
        bytes += used;
        for token in 0..n {
            mixed[token * channels + column..token * channels + column + rows]
                .copy_from_slice(&tile[token * rows..(token + 1) * rows]);
        }
        column += rows;
    }
    let (z, used) =
        crate::evaluate_integer_with_ax(&zp, &[], m, read, Some(&prep.q), Some(&prep.za))?;
    bytes += used;
    let mut window = history.to_vec();
    window.extend(mixed);
    let final_history = window[window.len() - 3 * channels..].to_vec();
    let mut cr = r.clone();
    cr.op = "conv_state".into();
    cr.tensor = format!("{root}.conv1d.weight");
    cr.aux.clear();
    let conv = m
        .tensors
        .iter()
        .find(|t| t.name == cr.tensor)
        .ok_or("head conv")?;
    let norm = m
        .tensors
        .iter()
        .find(|t| t.name == format!("{root}.norm.weight"))
        .ok_or("head norm")?;
    if conv.rows != 8192 || conv.cols != 4 || norm.rows.checked_mul(norm.cols) != Some(128) {
        return Err("head stage weights".into());
    }
    let mut weights = Vec::with_capacity(channels * 4);
    for (start, rows) in [
        (first / 2 * 128, h / 2 * 128),
        (2048 + first / 2 * 128, h / 2 * 128),
        (4096 + first * 128, h * 128),
    ] {
        cr.dims = vec![n, rows, 4, start];
        let (w, used) = crate::load_prepared_weight(conv, &cr, &mut *read)?;
        weights.extend_from_slice(&w);
        bytes += used;
    }
    cr.dims = vec![n, channels, 4, 0];
    let convolved = crate::execute(&cr, &window, &weights)?;
    drop(window);
    drop(weights);
    let (nw, used) = crate::load_prepared_weight(norm, r, &mut *read)?;
    bytes += used;
    let mut gated = vec![0.; n * h * 128];
    let mut qh = vec![];
    let mut kh = vec![];
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
        if head % 2 == 0 {
            qh = crate::execute(&nr, &gather(head / 2 * 128), &[])?;
            nr.scalars[1] = 128f32.sqrt().recip();
            kh = crate::execute(&nr, &gather(h / 2 * 128 + head / 2 * 128), &[])?;
        }
        let vh = gather(h * 128 + head * 128);
        let gh: Vec<_> = (0..n).map(|t| prep.gates[t * 32 + first + head]).collect();
        let bh: Vec<_> = (0..n)
            .map(|t| prep.gates[n * 32 + t * 32 + first + head])
            .collect();
        let mut values = crate::delta_from_key_major(
            &qh,
            &kh,
            &vh,
            &gh,
            &bh,
            &mut states[head * 16384..(head + 1) * 16384],
            128,
            128,
        )?;
        for value in &mut values {
            *value = crate::bf(*value);
        }
        for t in 0..n {
            values.extend_from_slice(&z[(t * h + head) * 128..(t * h + head + 1) * 128]);
        }
        nr.op = "gated_norm".into();
        nr.scalars = vec![1e-6];
        let y = crate::execute(&nr, &values, &nw)?;
        for t in 0..n {
            gated[(t * h + head) * 128..(t * h + head + 1) * 128]
                .copy_from_slice(&y[t * 128..(t + 1) * 128]);
        }
    }
    if !gated.iter().all(|v| v.is_finite()) {
        return Err("head group output".into());
    }
    Ok((gated, final_history, bytes))
}
