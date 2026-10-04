//! MLP completion, next Delta head group and out-projection columns in one query.
use crate::{
    prepared_weights::{LoadedWeight, WeightBuffer},
    Manifest, Request, Result,
};
pub(crate) const NAME: &str = "mlp-delta-stream-exact-v1";
fn is_follow(r: &Request) -> bool {matches!(r.op.as_str(), "delta_partial_mlp_prepare" | "delta_partial_mlp_prepare_down")}
const C: usize = 2560;
const H: usize = 9216;
const R: usize = 64;
fn metadata(r: &Request) -> Result<(usize, usize, usize, usize, usize, usize)> {
    if r.encoding != NAME
        || !matches!(
            r.op.as_str(),
            "mlp_complete_delta_partial" | "delta_partial_mlp_prepare" | "delta_partial_mlp_prepare_down"
        )
        || r.dims.len() != if r.op == "delta_partial_mlp_prepare_down" {6} else {5}
        || (r.op == "delta_partial_mlp_prepare_down" && !crate::mlp_pipeline::valid_partial_rows(*r.dims.get(5).unwrap_or(&0)))
        || r.aux.len() != 1
        || r.scalars.len() != 2
        || r.scalars[0].to_bits() != 2f32.to_bits()
        || r.scalars[1].to_bits() != 1e-6f32.to_bits()
    {
        return Err("pair metadata".into());
    }
    let (n, b, c, h, p) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3], r.dims[4]);
    if !(1..=89).contains(&n)
        || b == 0
        || c == 0
        || b % 256 != 0
        || c % 256 != 0
        || b.checked_add(c) != Some(H)
        || h == 0
        || h >= 32
        || h % 2 != 0
        || p == 0
        || p > 132
    {
        return Err("pair bounds".into());
    }
    let s = r
        .tensor
        .strip_prefix("model.language_model.layers.")
        .and_then(|v| v.strip_suffix(".post_attention_layernorm.weight"))
        .ok_or("pair layer")?;
    let layer: usize = s.parse().map_err(|_| "pair layer")?;
    if layer >= 30
        || layer.to_string() != s
        || (layer + 2) % 4 == 0
        || r.aux[0]
            != format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            )
    {
        return Err("pair next Delta scope".into());
    }
    Ok((n, b, c, h, p, layer))
}
fn inner(r: &Request) -> Result<Request> {
    let (n, b, c, _, _, _) = metadata(r)?;
    let mut i = r.clone();
    i.encoding = crate::mlp_stream::NAME.into();
    i.op = "mlp_stream_complete".into();
    i.dims = vec![n, b, c];
    Ok(i)
}
pub(crate) fn reply_count(r: &Request) -> Result<usize> {
    let (n, _, _, h, _, _) = metadata(r)?;
    Ok(if is_follow(r) {
        n * (C + H + 100) + 3 * (32 - h) * 256
    } else {
        n * (2 * C + C / 256 + 2 * R + 64 + C + R) + 3 * h * 256
    })
}
/// Validated complete MLP input remains attached to the whole fusion request.
/// ```compile_fail
/// let mut x: imajev_runtime::PreparedMlpDeltaStream = todo!();
/// x.request.step += 1;
/// ```
pub struct PreparedMlpDeltaStream {
    request: Request,
    mlp_request: Request,
    mlp: Option<crate::PreparedMlpStream>,
    follow: Option<(
        Vec<f32>,
        crate::delta_head_continue::Preparation,
        Vec<f32>,
        Vec<f32>,
    )>,
    history: Vec<f32>,
    log: Vec<f32>,
}
impl PreparedMlpDeltaStream {
    pub(crate) fn decode(r: &Request, payload: &[u8]) -> Result<Self> {
        let (n, _, _, h, p, _) = metadata(r)?;
        if is_follow(r) {
            return Self::decode_follow(r, payload, n, h, p);
        }
        if payload.first() != Some(&1) || payload.len() < 5 {
            return Err("pair request direction".into());
        }
        let len = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
        let end = 5usize
            .checked_add(len)
            .filter(|end| *end <= payload.len())
            .ok_or("pair MLP length")?;
        let hc = 3 * h * 256;
        let kc = p * h / 2 * 128;
        let tail = p * (h * 128 + h);
        if payload.len() != end + 2 * (hc + kc) + 4 * tail {
            return Err("pair prefix length".into());
        }
        let mlp_request = inner(r)?;
        let mlp = crate::PreparedMlpStream::decode(&mlp_request, &payload[5..end])?;
        let mut history = vec![0.; hc];
        crate::bf16_codec::unpack(&payload[end..end + 2 * hc], &mut history);
        let mut log = vec![0.; kc];
        crate::bf16_codec::unpack(&payload[end + 2 * hc..end + 2 * (hc + kc)], &mut log);
        log.extend(
            payload[end + 2 * (hc + kc)..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
        if !history.iter().chain(&log).all(|v| v.is_finite())
            || !log[kc + p * h * 128..]
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        {
            return Err("pair prefix finite/decay".into());
        }
        Ok(Self {
            request: r.clone(),
            mlp_request,
            mlp: Some(mlp),
            follow: None,
            history,
            log,
        })
    }
    fn decode_follow(r: &Request, payload: &[u8], n: usize, h: usize, p: usize) -> Result<Self> {
        let remaining = 32 - h;
        let hc = 3 * remaining * 256;
        let kc = p * remaining / 2 * 128;
        let prefix_tail = p * (remaining * 128 + remaining);
        let tail = n * (C / 256 + 2 * R + 64 + C + R);
        let end = 1 + 3 * n * C + 4 * tail;
        if payload.first() != Some(&2) || payload.len() != end + 2 * (hc + kc) + 4 * prefix_tail {
            return Err("pair continuation length/direction".into());
        }
        let mut hidden = vec![0.; n * C];
        crate::bf16_codec::unpack(&payload[1..1 + 2 * n * C], &mut hidden);
        let ints = &payload[1 + 2 * n * C..1 + 3 * n * C];
        let rest: Vec<_> = payload[1 + 3 * n * C..end]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if !hidden.iter().chain(&rest).all(|v| v.is_finite()) {
            return Err("pair continuation finite".into());
        }
        let prep_end = n * (C / 256 + 2 * R + 64);
        let prep = crate::delta_head_continue::Preparation::restore(n, ints, &rest[..prep_end])?;
        let base = rest[prep_end..prep_end + n * C].to_vec();
        let ax = rest[prep_end + n * C..].to_vec();
        let mut history = vec![0.; hc];
        crate::bf16_codec::unpack(&payload[end..end + 2 * hc], &mut history);
        let mut log = vec![0.; kc];
        crate::bf16_codec::unpack(&payload[end + 2 * hc..end + 2 * (hc + kc)], &mut log);
        log.extend(
            payload[end + 2 * (hc + kc)..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
        if !history.iter().chain(&log).all(|v| v.is_finite())
            || !log[kc + p * remaining * 128..]
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        {
            return Err("pair continuation prefix".into());
        }
        Ok(Self {
            request: r.clone(),
            mlp_request: inner(r)?,
            mlp: None,
            follow: Some((hidden, prep, base, ax)),
            history,
            log,
        })
    }
    fn evaluate_follow<F, B>(
        self,
        r: &Request,
        m: &Manifest,
        read: &mut F,
        n: usize,
        h: usize,
        p: usize,
        layer: usize,
    ) -> Result<(Vec<f32>, u64)>
    where
        F: FnMut(u64, usize) -> Result<B>,
        B: WeightBuffer,
    {
        let (hidden, prep, base, ax) = self.follow.ok_or("pair continuation direction")?;
        let remaining = 32 - h;
        let count = remaining * 128;
        let mut dr = r.clone();
        dr.op = "delta_project_reuse".into();
        dr.encoding = "bf16-block256-exact-v1".into();
        dr.tensor = format!(
            "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",
            layer + 1
        );
        dr.aux.clear();
        dr.scalars.clear();
        dr.dims = vec![n, remaining, h, 0];
        let (gated, history, mut bytes) = crate::profile::measure("pair_remaining_heads", || {
            crate::delta_head_continue::group(
                &dr,
                &prep,
                &self.history,
                &self.log,
                n,
                remaining,
                h,
                p,
                m,
                read,
            )
        })?;
        let root = dr.tensor.strip_suffix(".in_proj_qkv.weight").unwrap();
        let mut kr = dr.clone();
        kr.op = "linear_integer_k_continue".into();
        kr.tensor = format!("{root}.out_proj.weight");
        kr.dims = vec![n, C, 4096, h * 128, count];
        let mut input = gated.clone();
        input.extend(base);
        let (base, used) = crate::int8_k_continue::evaluate(&kr, &input, m, read)?;
        bytes += used;
        drop(input);
        let aname = format!("{root}.out_proj.lora_A.weight");
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == aname)
            .ok_or("pair continuation A")?;
        if a.dtype != "f32" || a.rows != R || a.cols != 4096 || a.bytes != (R * 4096 * 4) as u64 {
            return Err("pair continuation A shape".into());
        }
        kr.op = "matmul".into();
        kr.tensor = aname;
        kr.dims = vec![n, R, 4096];
        let (w, used) = crate::load_prepared_weight(a, &kr, &mut *read)?;
        bytes += used;
        let LoadedWeight::Prepared(w) = w else {
            return Err("pair continuation fixed A".into());
        };
        let ax = crate::f32_output::continue_columns(&gated, &ax, &w, n, R, 4096, h * 128, count)?;
        drop(w);
        kr.op = "linear_integer_k_finish".into();
        kr.tensor = format!("{root}.out_proj.weight");
        kr.dims = vec![n, C, 4096];
        kr.scalars = vec![2.];
        kr.aux = vec![
            format!("{root}.out_proj.lora_A.weight"),
            format!("{root}.out_proj.lora_B.weight"),
        ];
        let mut state = base;
        state.extend(ax);
        let (attention, used) = crate::int8_k_continue::evaluate(&kr, &state, m, read)?;
        bytes += used;
        drop(state);
        let mr = next_mlp(r)?;
        let mut pair = hidden;
        pair.extend(attention);
        let (mut out, used) = crate::profile::measure("pair_next_mlp_prepare", || {
            if mr.op == "mlp_prepare_partial_down" {
                crate::mlp_pipeline::prepare_partial(&mr, &pair, m, read)
            } else {crate::mlp_pipeline::prepare(&mr, &pair, m, read)}
        })?;
        bytes += used;
        out.extend(history);
        Ok((out, bytes))
    }
    pub(crate) fn evaluate<F, B>(
        self,
        r: &Request,
        m: &Manifest,
        read: &mut F,
    ) -> Result<(Vec<f32>, u64)>
    where
        F: FnMut(u64, usize) -> Result<B>,
        B: WeightBuffer,
    {
        if !crate::same_request(r, &self.request) {
            return Err("pair identity".into());
        }
        let (n, _, _, h, p, layer) = metadata(r)?;
        if is_follow(r) {
            return self.evaluate_follow(r, m, read, n, h, p, layer);
        }
        let (both, mut bytes) = crate::profile::measure("pair_mlp_complete", || {
            self.mlp
                .ok_or("pair completion direction")?
                .evaluate(&self.mlp_request, m, read)
        })?;
        let mut dr = r.clone();
        dr.op = "delta_project_capture".into();
        dr.encoding = "bf16-block256-exact-v1".into();
        dr.tensor = format!(
            "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",
            layer + 1
        );
        dr.aux.clear();
        dr.scalars.clear();
        dr.dims = vec![n, h, 0, 0];
        let (prep, used) =
            crate::delta_head_continue::Preparation::capture(&dr, &both[n * C..], m, read)?;
        bytes += used;
        let (gated, history, used) = crate::profile::measure("pair_delta_heads", || {
            crate::delta_head_continue::group(
                &dr,
                &prep,
                &self.history,
                &self.log,
                n,
                h,
                0,
                p,
                m,
                read,
            )
        })?;
        bytes += used;
        let root = dr.tensor.strip_suffix(".in_proj_qkv.weight").unwrap();
        let count = h * 128;
        let mut kr = r.clone();
        kr.encoding = "bf16-block256-exact-v1".into();
        kr.op = "linear_integer_k_continue".into();
        kr.tensor = format!("{root}.out_proj.weight");
        kr.aux.clear();
        kr.scalars.clear();
        kr.dims = vec![n, C, 4096, 0, count];
        let mut input = gated.clone();
        input.extend(vec![0.; n * C]);
        let (base, used) = crate::int8_k_continue::evaluate(&kr, &input, m, read)?;
        bytes += used;
        drop(input);
        let aname = format!("{root}.out_proj.lora_A.weight");
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == aname)
            .ok_or("pair out A")?;
        if a.dtype != "f32" || a.rows != R || a.cols != 4096 || a.bytes != (R * 4096 * 4) as u64 {
            return Err("pair out A shape".into());
        }
        kr.op = "matmul".into();
        kr.tensor = aname;
        kr.dims = vec![n, R, 4096];
        let (w, used) = crate::load_prepared_weight(a, &kr, &mut *read)?;
        bytes += used;
        let LoadedWeight::Prepared(w) = w else {
            return Err("pair fixed F32 A".into());
        };
        let ax = crate::profile::measure("pair_out_A_continue", || {
            crate::f32_output::continue_columns(&gated, &vec![0.; n * R], &w, n, R, 4096, 0, count)
        })?;
        let mut out = both[..n * C].to_vec();
        prep.flat(n, &mut out);
        out.extend(base);
        out.extend(ax);
        out.extend(history);
        if out.len() != reply_count(r)? || !out.iter().all(|v| v.is_finite()) {
            return Err("pair output length/finite".into());
        }
        Ok((out, bytes))
    }
}
fn next_mlp(r: &Request) -> Result<Request> {
    let (n, _, _, _, _, layer) = metadata(r)?;
    let mut mr = r.clone();
    mr.op = "mlp_prepare_down".into();
    mr.encoding = crate::mlp_pipeline::NAME.into();
    mr.tensor = format!(
        "model.language_model.layers.{}.post_attention_layernorm.weight",
        layer + 1
    );
    mr.aux = vec![format!(
        "model.language_model.layers.{}.input_layernorm.weight",
        layer + 2
    )];
    mr.dims = vec![n, C];
    if r.op == "delta_partial_mlp_prepare_down" {
        mr.op = "mlp_prepare_partial_down".into();
        mr.dims.push(r.dims[5]);
    }
    Ok(mr)
}
fn pack_bf(b: &mut Vec<u8>, x: &[f32]) -> Result<()> {
    if !crate::bf16_codec::all_bf16(x) {
        return Err("pair BF16 reply".into());
    }
    let start = b.len();
    b.resize(start + 2 * x.len(), 0);
    crate::bf16_codec::pack(x, &mut b[start..]);
    Ok(())
}
pub(crate) fn append_reply(b: &mut Vec<u8>, r: &Request, x: &[f32]) -> Result<()> {
    let (n, _, _, h, _, _) = metadata(r)?;
    if x.len() != reply_count(r)? || !x.iter().all(|v| v.is_finite()) {
        return Err("pair reply length/finite".into());
    }
    if is_follow(r) {
        b.push(3);
        let c = n * (C + H + 100);
        crate::mlp_pipeline::append(b, &next_mlp(r)?, &x[..c])?;
        return pack_bf(b, &x[c..]);
    }
    let mut cursor = 0;
    b.push(0);
    fn bf(b: &mut Vec<u8>, x: &[f32]) -> Result<()> {
        if !crate::bf16_codec::all_bf16(x) {
            return Err("pair BF16 reply".into());
        }
        let start = b.len();
        b.resize(start + 2 * x.len(), 0);
        crate::bf16_codec::pack(x, &mut b[start..]);
        Ok(())
    }
    bf(b, &x[..n * C])?;
    cursor += n * C;
    let start = b.len();
    b.resize(start + n * C, 0);
    crate::projection_codec::pack_integers(&x[cursor..cursor + n * C], &mut b[start..])?;
    cursor += n * C;
    let tail = n * (C / 256 + 2 * R + 64 + C + R);
    if x[cursor..cursor + n * C / 256].iter().any(|v| *v <= 0.)
        || x[cursor + n * (C / 256 + 2 * R)..cursor + n * (C / 256 + 2 * R + 64)]
            .iter()
            .any(|v| !(0.0..=1.0).contains(v))
    {
        return Err("pair reply scales/gates".into());
    }
    for v in &x[cursor..cursor + tail] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    cursor += tail;
    bf(b, &x[cursor..cursor + 3 * h * 256])?;
    Ok(())
}
pub(crate) fn decode_reply(r: &Request, payload: &[u8]) -> Result<Vec<f32>> {
    let (n, _, _, h, _, _) = metadata(r)?;
    if is_follow(r) {
        let end = 2 + n * (C * 2 + H + 400);
        let hc = 3 * (32 - h) * 256;
        if payload.first() != Some(&3) || payload.len() != end + 2 * hc {
            return Err("pair prepared reply direction/length".into());
        }
        let mut out = crate::mlp_pipeline::decode_values(&next_mlp(r)?, &payload[1..end])?;
        let mut history = vec![0.; hc];
        crate::bf16_codec::unpack(&payload[end..], &mut history);
        if !history.iter().all(|v| v.is_finite()) {
            return Err("pair history finite".into());
        }
        out.extend(history);
        return Ok(out);
    }
    let bf_count = n * C;
    let tail = n * (C / 256 + 2 * R + 64 + C + R);
    if payload.first() != Some(&0) || payload.len() != 1 + bf_count * 3 + tail * 4 + 3 * h * 256 * 2
    {
        return Err("pair reply length/direction".into());
    }
    let mut out = vec![0.; bf_count];
    crate::bf16_codec::unpack(&payload[1..1 + bf_count * 2], &mut out);
    let mut cursor = 1 + bf_count * 2;
    if payload[cursor..cursor + bf_count].contains(&128) {
        return Err("pair integer reply".into());
    }
    out.extend(
        payload[cursor..cursor + bf_count]
            .iter()
            .map(|b| *b as i8 as f32),
    );
    cursor += bf_count;
    out.extend(
        payload[cursor..cursor + tail * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
    );
    cursor += tail * 4;
    let mut history = vec![0.; 3 * h * 256];
    crate::bf16_codec::unpack(&payload[cursor..], &mut history);
    out.extend(history);
    let mut checked = vec![];
    append_reply(&mut checked, r, &out)?;
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request(n: usize, h: usize) -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_complete_delta_partial","encoding":NAME,"tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[n,4352,4864,h,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"]})).unwrap()
    }
    #[test]
    fn maximum_reply_is_exact_bounded_and_not_a_query_input() {
        for n in [1, 7, 87, 89] {
            for h in [22, 24] {
                let r = request(n, h);
                let mut x = vec![-0.; n * C];
                x.extend(vec![-127.; n * C]);
                x.extend(vec![0.0123; n * C / 256]);
                x.extend(vec![0.1234567; n * 2 * R]);
                x.extend(vec![0.875; n * 64]);
                x.extend(vec![-0.234567; n * C]);
                x.extend(vec![0.345678; n * R]);
                x.extend(vec![0.5; 3 * h * 256]);
                let frame = crate::encode(&r, &x).unwrap();
                assert!(frame.len() < 2_000_000);
                let (_, y) = crate::decode(&frame).unwrap();
                assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(crate::decode_query(&frame).is_err());
                for offset in [2 * n * C, 2 * n * C + n * (C / 256 + 2 * R)] {
                    let mut bad = x.clone();
                    bad[offset] = f32::NAN;
                    assert!(crate::encode(&r, &bad).is_err());
                }
            }
        }
    }
    fn input(r: &Request) -> Vec<u8> {
        let (n, b, _, h, p, _) = metadata(r).unwrap();
        let mr = inner(r).unwrap();
        let mut x = vec![0.; n * 2 * C];
        x.extend(vec![1.; n * C / 256]);
        x.extend(vec![0.; n * 2 * R]);
        x.extend(vec![0.; n * b]);
        x.extend(vec![1.; n * b / 256]);
        x.extend(vec![0.; n * R]);
        let frame = crate::encode(&mr, &x).unwrap();
        let hlen = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        let part = &frame[4 + hlen..frame.len() - 32];
        let mut payload = vec![1];
        payload.extend((part.len() as u32).to_le_bytes());
        payload.extend(part);
        payload.extend(vec![0u8; 2 * (3 * h * 256 + p * h / 2 * 128)]);
        for _ in 0..p * (h * 128 + h) {
            payload.extend(0f32.to_le_bytes());
        }
        payload
    }
    #[test]
    fn continuation_direction_identity_and_large_prepared_reply_are_checked() {
        for n in [1, 89] {
            let mut r = request(n, 22);
            let mut x = vec![0.; n * 2 * C];
            x.extend(vec![1.; n * C / 256]);
            x.extend(vec![0.1234567; n * 2 * R]);
            x.extend(vec![0.875; n * 64]);
            x.extend(vec![0.234567; n * C]);
            x.extend(vec![0.345678; n * R]);
            x.extend(vec![0.; 3 * 22 * 256]);
            let mut encoded = vec![];
            append_reply(&mut encoded, &r, &x).unwrap();
            encoded.truncate(encoded.len() - 2 * 3 * 22 * 256);
            encoded[0] = 2;
            let remaining = 10;
            encoded.extend(vec![
                0;
                2 * (3 * remaining * 256 + 45 * remaining / 2 * 128)
            ]);
            for _ in 0..45 * (remaining * 128 + remaining) {
                encoded.extend(0f32.to_le_bytes());
            }
            r.op = "delta_partial_mlp_prepare".into();
            let header_bytes = serde_json::to_vec(&r).unwrap().len();
            assert!(encoded.len() + header_bytes + 36 < 2_000_000);
            let prepared = PreparedMlpDeltaStream::decode(&r, &encoded).unwrap();
            let m = Manifest {
                version: 1,
                model: r.model.clone(),
                pack_hash: r.pack_hash.clone(),
                bytes: 0,
                tensors: vec![],
            };
            let mut bad = r.clone();
            bad.step += 1;
            assert_eq!(
                prepared
                    .evaluate(&bad, &m, &mut |_, _| -> Result<Vec<u8>> {
                        panic!("follow identity read")
                    })
                    .unwrap_err(),
                "pair identity"
            );
            let mut invalid = encoded.clone();
            invalid[1 + 2 * n * C] = 128;
            assert!(PreparedMlpDeltaStream::decode(&r, &invalid).is_err());
            invalid = encoded.clone();
            invalid[1 + 3 * n * C..1 + 3 * n * C + 4].copy_from_slice(&0f32.to_le_bytes());
            assert!(PreparedMlpDeltaStream::decode(&r, &invalid).is_err());
            let mut reply = vec![-0.; n * C];
            reply.extend(vec![127.; n * H]);
            reply.extend(vec![0.009; n * 36]);
            reply.extend(vec![-0.1234567; n * R]);
            reply.extend(vec![0.; 3 * remaining * 256]);
            let frame = crate::encode(&r, &reply).unwrap();
            assert!(frame.len() < 2_000_000);
            let (_, actual) = crate::decode(&frame).unwrap();
            assert!(reply
                .iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(crate::decode_query(&frame).is_err());
        }
    }
    #[test]
    fn partial_down_progress_is_explicit_and_bound() {
        let mut r = request(1, 22);
        r.op = "delta_partial_mlp_prepare_down".into();
        r.dims.push(1600);
        assert!(metadata(&r).is_ok());
        let next = next_mlp(&r).unwrap();
        assert_eq!(next.op, "mlp_prepare_partial_down");
        assert_eq!(next.dims, vec![1, C, 1600]);
        for rows in [0, 31, C, usize::MAX] {
            r.dims[5] = rows;
            assert!(metadata(&r).is_err());
        }
        r.dims[5] = 1600;
        r.op = "delta_partial_mlp_prepare".into();
        assert!(metadata(&r).is_err());
    }
    #[test]
    fn malformed_metadata_prefix_and_identity_fail_before_reads() {
        let r = request(1, 24);
        let payload = input(&r);
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![],
        };
        let state = PreparedMlpDeltaStream::decode(&r, &payload).unwrap();
        let mut changed = r.clone();
        changed.step += 1;
        assert_eq!(
            state
                .evaluate(&changed, &m, &mut |_, _| -> Result<Vec<u8>> {
                    panic!("identity read")
                })
                .unwrap_err(),
            "pair identity"
        );
        for dims in [
            vec![],
            vec![0, 4352, 4864, 24, 45],
            vec![90, 4352, 4864, 24, 45],
            vec![1, 0, 9216, 24, 45],
            vec![1, 4352, 4864, 23, 45],
            vec![1, 4352, 4864, 32, 45],
            vec![1, 4352, 4864, 24, 0],
            vec![1, 4352, 4864, 24, 133],
            vec![1, usize::MAX, 256, 24, 45],
        ] {
            let mut bad = r.clone();
            bad.dims = dims;
            assert!(PreparedMlpDeltaStream::decode(&bad, &payload).is_err());
        }
        for value in [f32::NAN, 1.1, -1.] {
            let mut bad = payload.clone();
            let last = bad.len() - 4;
            bad[last..].copy_from_slice(&value.to_le_bytes());
            assert!(PreparedMlpDeltaStream::decode(&r, &bad).is_err());
        }
        let mut bad = payload.clone();
        bad.push(0);
        assert!(PreparedMlpDeltaStream::decode(&r, &bad).is_err());
        assert!(PreparedMlpDeltaStream::decode(&r, &payload[..payload.len() - 1]).is_err());
    }
}
