//! Original-order block256 accumulation. Only new gated columns are quantized.
use crate::{prepared_weights::WeightBuffer, Manifest, Request, Result};
fn shape(r: &Request) -> Result<(usize, usize, usize, usize, usize)> {
    if r.op == "linear_integer_k_finish" {
        let prefix = r
            .tensor
            .strip_suffix(".weight")
            .filter(|s| s.ends_with(".linear_attn.out_proj"))
            .ok_or("INT8 finish tensor")?;
        if !matches!(r.encoding.as_str(), "bf16-exact" | "bf16-block256-exact-v1")
            || r.dims.len() != 3
            || r.dims[0] == 0
            || r.dims[0] > 89
            || r.dims[1] != 2560
            || r.dims[2] != 4096
            || r.scalars.len() != 1
            || r.scalars[0].to_bits() != 2f32.to_bits()
            || r.aux
                != [
                    format!("{prefix}.lora_A.weight"),
                    format!("{prefix}.lora_B.weight"),
                ]
        {
            return Err("INT8 finish metadata/bounds".into());
        }
        return Ok((r.dims[0], 2560, 4096, 4096, 0));
    }
    if r.op != "linear_integer_k_continue"
        || r.dims.len() != 5
        || !r.aux.is_empty()
        || !r.scalars.is_empty()
        || !matches!(r.encoding.as_str(), "bf16-exact" | "bf16-block256-exact-v1")
        || !r.tensor.ends_with(".linear_attn.out_proj.weight")
    {
        return Err("INT8 continuation metadata".into());
    }
    let (n, rows, cols, begin, count) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3], r.dims[4]);
    if n == 0
        || n > 89
        || rows != 2560
        || cols != 4096
        || begin % 256 != 0
        || count == 0
        || count % 256 != 0
        || begin.checked_add(count).is_none_or(|v| v > cols)
        || n.checked_mul(count + rows)
            .is_none_or(|v| v > crate::MAX_FLOATS)
    {
        return Err("INT8 continuation bounds".into());
    }
    Ok((n, rows, cols, begin, count))
}
fn input(r: &Request, x: &[f32]) -> Result<()> {
    let (n, rows, _, begin, count) = shape(r)?;
    if r.op == "linear_integer_k_finish" {
        return if x.len() == n * (rows + 64) && x.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err("INT8 finish finite state/shape".into())
        };
    }
    if x.len() != n * (count + rows)
        || !x.iter().all(|v| v.is_finite())
        || !crate::bf16_codec::all_bf16(&x[..n * count])
        || begin == 0 && x[n * count..].iter().any(|v| v.to_bits() != 0)
    {
        return Err("INT8 continuation input/initial precision".into());
    }
    Ok(())
}
/// Checked input and accumulator stay attached to the complete request.
/// ```compile_fail
/// let mut state: imajev_runtime::PreparedInt8K = todo!();
/// state.values.clear();
/// ```
pub struct PreparedInt8K {
    request: Request,
    values: Vec<f32>,
}
impl PreparedInt8K {
    pub(crate) fn decode(r: &Request, p: &[u8]) -> Result<Self> {
        shape(r)?;
        let values = crate::decode_values(r, p)?;
        input(r, &values)?;
        Ok(Self {
            request: r.clone(),
            values,
        })
    }
    pub(crate) fn evaluate<F, B>(
        &self,
        r: &Request,
        m: &Manifest,
        read: &mut F,
    ) -> Result<(Vec<f32>, u64)>
    where
        F: FnMut(u64, usize) -> Result<B>,
        B: WeightBuffer,
    {
        if !crate::same_request(r, &self.request) {
            return Err("INT8 continuation identity".into());
        }
        core(r, &self.values, m, read)
    }
}
pub(crate) fn evaluate<F, B>(
    r: &Request,
    x: &[f32],
    m: &Manifest,
    read: &mut F,
) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: WeightBuffer,
{
    input(r, x)?;
    core(r, x, m, read)
}
fn core<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: WeightBuffer,
{
    let (n, rows, cols, begin, count) = shape(r)?;
    let t = m
        .tensors
        .iter()
        .find(|t| t.name == r.tensor)
        .ok_or("INT8 continuation tensor")?;
    if t.dtype != "int8"
        || t.rows != rows
        || t.cols != cols
        || t.bytes != (rows * cols + rows * 4) as u64
    {
        return Err("INT8 continuation weight shape".into());
    }
    if r.op == "linear_integer_k_finish" {
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[0])
            .ok_or("INT8 finish A metadata")?;
        let b = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[1])
            .ok_or("INT8 finish B metadata")?;
        if a.dtype != "f32"
            || a.rows != 64
            || a.cols != cols
            || a.bytes != (64 * cols * 4) as u64
            || b.dtype != "f32"
            || b.rows != rows
            || b.cols != 64
            || b.bytes != (rows * 64 * 4) as u64
        {
            return Err("INT8 finish adapter shape".into());
        }
        let mut br = r.clone();
        br.op = "matmul".into();
        br.tensor = b.name.clone();
        br.dims = vec![n, rows, 64];
        br.aux.clear();
        br.scalars.clear();
        let (w, bytes) = crate::load_prepared_weight(b, &br, &mut *read)?;
        if !matches!(w, crate::prepared_weights::LoadedWeight::Prepared(_)) {
            return Err("INT8 finish fixed F32 B required".into());
        }
        let z = crate::profile::measure("integer_k_finish_B", || {
            crate::matrix_loaded(&x[n * rows..], &w, n, rows, 64)
        })?;
        let y: Vec<_> = x[..n * rows]
            .iter()
            .zip(z)
            .map(|(v, z)| crate::bf(crate::bf(*v) + crate::bf(2. * z)))
            .collect();
        if !y.iter().all(|v| v.is_finite()) {
            return Err("INT8 finish output finite".into());
        }
        return Ok((y, bytes));
    }
    let raw = read(t.offset, rows * cols)?;
    let w = raw
        .prepared_output_pairs()
        .ok_or("INT8 continuation fixed raw S1 required")?;
    if w.rows() != rows || w.cols() != cols {
        return Err("INT8 continuation prepared shape".into());
    }
    let q = crate::profile::measure("integer_k_quantize_new_columns", || {
        crate::int8_kernel::quantize_rows(&x[..n * count], n, count)
    })?;
    let q = crate::profile::measure("integer_k_place_columns", || q.place_columns(cols, begin))?;
    let y = crate::profile::measure("integer_k_continue", || {
        crate::output_pairs::project_continued(&q, &w, begin, count, &x[n * count..])
    })?;
    Ok((y, t.bytes))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"linear_integer_k_continue","tensor":"model.language_model.layers.0.linear_attn.out_proj.weight","dims":[1,2560,4096,0,256],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap()
    }
    #[test]
    fn bounds_and_bad_state_fail_before_reads() {
        let r = request();
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![],
        };
        let mut read = |_, _| -> Result<Vec<u8>> { panic!("invalid read") };
        for dims in [
            vec![],
            vec![0, 2560, 4096, 0, 256],
            vec![90, 2560, 4096, 0, 256],
            vec![1, 2560, 4096, 1, 256],
            vec![1, 2560, 4096, 4096, 256],
            vec![1, 2560, 4096, 0, usize::MAX],
        ] {
            let mut bad = r.clone();
            bad.dims = dims;
            assert!(evaluate(&bad, &[], &m, &mut read).is_err());
        }
        for v in [-0., 1., f32::NAN] {
            let mut x = vec![0.; 256 + 2560];
            x[256] = v;
            assert!(evaluate(&r, &x, &m, &mut read).is_err());
        }
        let mut x = vec![0.; 256 + 2560];
        x[0] = 0.1234567;
        assert!(evaluate(&r, &x, &m, &mut read).is_err());
    }
    #[test]
    fn typed_progress_rejects_before_reads() {
        let r = request();
        let frame = crate::encode(&r, &vec![0.; 256 + 2560]).unwrap();
        let (_, state) = crate::decode_query(&frame).unwrap();
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![],
        };
        let mut bad = r.clone();
        bad.step += 1;
        assert!(crate::evaluate_decoded_with_prepared_buffer(
            &bad,
            &state,
            &m,
            |_, _| -> Result<Vec<u8>> { panic!("identity read") }
        )
        .is_err());
    }
    #[test]
    fn independent_chunks_preserve_full_base_bits_with_views_and_odd_tokens() {
        use crate::{
            int8_kernel,
            output_pairs::{project_continued, project_prepared, PreparedPairs},
        };
        for cols in [512, 4096] {
            let rows = 512;
            let mut bytes: Vec<u8> = (0..rows * cols)
                .map(|i| ((i * 17 + 13) % 256) as u8)
                .collect();
            let scales: Vec<f32> = (0..rows)
                .map(|i| 0.000127 + (i % 17) as f32 * 0.00003)
                .collect();
            for s in &scales {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            let fixed = PreparedPairs::from_le_bytes(&bytes, rows, cols).unwrap();
            for (n, start, width) in [(1, 0, 128), (7, 96, 32), (87, 64, 128)] {
                let view = fixed.packed_rows(start, width).unwrap();
                let x: Vec<f32> = (0..n * cols)
                    .map(|i| crate::bf([0., -0., 0.1234567, -0.91, f32::MIN_POSITIVE, 17.][i % 6]))
                    .collect();
                let all = int8_kernel::quantize_rows(&x, n, cols).unwrap();
                let expected = project_prepared(&all, &view).unwrap();
                let w: Vec<i8> = bytes[start * cols..(start + width) * cols]
                    .iter()
                    .map(|v| *v as i8)
                    .collect();
                let reference =
                    int8_kernel::project(&all, &w, &scales[start..start + width], width).unwrap();
                assert!(expected
                    .iter()
                    .zip(reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
                for chunk in [256, 512, 1024, 2816] {
                    let mut out = vec![0.; n * width];
                    let mut begin = 0;
                    while begin < cols {
                        let count = chunk.min(cols - begin);
                        let mut values = vec![];
                        for token in 0..n {
                            values.extend_from_slice(
                                &x[token * cols + begin..token * cols + begin + count],
                            );
                        }
                        let q = int8_kernel::quantize_rows(&values, n, count)
                            .unwrap()
                            .place_columns(cols, begin)
                            .unwrap();
                        out = project_continued(&q, &view, begin, count, &out).unwrap();
                        begin += count;
                    }
                    assert!(
                        out.iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "cols={cols} n={n} start={start} chunk={chunk}"
                    );
                }
            }
        }
    }
}
