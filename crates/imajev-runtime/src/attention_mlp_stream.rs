//! Client-held bridge across MLP completion, full-attention KV/Q and next MLP.
use crate::{
    int8_kernel::QuantizedRows, prepared_weights::WeightBuffer, Manifest, Request, Result,
};
pub(crate) const NAME: &str = "attention-mlp-stream-exact-v1";
const C: usize = 2560;
const KV: usize = 2048;
fn complete(r: &Request) -> bool {
    matches!(
        r.op.as_str(),
        "mlp_complete_attention_kv" | "mlp_complete_attention_kv_q4"
    )
}
fn q4(r: &Request) -> bool {
    matches!(
        r.op.as_str(),
        "mlp_complete_attention_kv_q4" | "attention_finish_mlp_front_q4" | "attention_finish_mlp_front_q4_compact"
    )
}
fn compact(r:&Request)->bool {r.op=="attention_finish_mlp_front_q4_compact"}
fn q_payload(
    norm: &[f32],
    kv: &[f32],
    prefix: &[f32],
    n: usize,
    p: usize,
    first: usize,
    groups: usize,
) -> Vec<f32> {
    let mut out = norm.to_vec();
    for part in 0..2 {
        for head in first..first + groups {
            let start = part * p * 1024 + head * p * 256;
            out.extend_from_slice(&prefix[start..start + p * 256]);
            for t in 0..n {
                let start = part * n * 1024 + (t * 4 + head) * 256;
                out.extend_from_slice(&kv[start..start + 256]);
            }
        }
    }
    out
}
fn unpack_bf(p: &[u8]) -> Result<Vec<f32>> {
    let mut v = vec![0.; p.len() / 2];
    crate::bf16_codec::unpack(p, &mut v);
    if !v.iter().all(|v| v.is_finite()) {
        return Err("attention MLP carry finite".into());
    }
    Ok(v)
}
fn unpack_f32(p: &[u8]) -> Result<Vec<f32>> {
    let v: Vec<_> = p
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if p.len() % 4 != 0 || !v.iter().all(|v| v.is_finite()) {
        return Err("attention MLP F32 finite".into());
    }
    Ok(v)
}
fn append_bf(b: &mut Vec<u8>, v: &[f32]) -> Result<()> {
    if !crate::bf16_codec::all_bf16(v) {
        return Err("attention MLP BF16 reply".into());
    }
    let pos = b.len();
    b.resize(pos + 2 * v.len(), 0);
    crate::bf16_codec::pack(v, &mut b[pos..]);
    Ok(())
}
fn metadata(r: &Request) -> Result<(usize, usize, usize, usize)> {
    let complete = complete(r);
    if r.encoding != NAME
        || !(complete
            || matches!(
                r.op.as_str(),
                "attention_finish_mlp_front" | "attention_finish_mlp_front_q4" | "attention_finish_mlp_front_q4_compact"
            ))
        || r.dims.len() != if complete { 4 } else { 3 }
        || r.aux.len() != 1
        || r.scalars.len() != 2
        || r.scalars[0].to_bits() != 2f32.to_bits()
        || r.scalars[1].to_bits() != 1e-6f32.to_bits()
    {
        return Err("attention MLP metadata".into());
    }
    let n = r.dims[0];
    let b = r.dims[1];
    let p = r.dims[if complete { 3 } else { 2 }];
    if !(1..=89).contains(&n)
        || b == 0
        || b >= 9216
        || b % crate::mlp_stream::STEP != 0
        || p > 132
        || n + p > 512
        || complete && b.checked_add(r.dims[2]) != Some(9216)
    {
        return Err("attention MLP bounds".into());
    }
    let s = r
        .tensor
        .strip_prefix("model.language_model.layers.")
        .and_then(|s| s.strip_suffix(".post_attention_layernorm.weight"))
        .ok_or("attention MLP tensor")?;
    let layer: usize = s.parse().map_err(|_| "attention MLP layer")?;
    if layer.to_string() != s
        || layer >= 30
        || layer % 4 != if complete { 2 } else { 3 }
        || r.aux[0]
            != format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            )
    {
        return Err("attention MLP scope".into());
    }
    Ok((n, b, p, layer))
}
fn mlp(r: &Request) -> Result<Request> {
    let (n, b, _, _) = metadata(r)?;
    let mut inner = r.clone();
    inner.encoding = crate::mlp_stream::NAME.into();
    inner.op = if complete(r) {
        "mlp_stream_complete"
    } else {
        "mlp_stream_prepare"
    }
    .into();
    inner.dims = if complete(r) {
        vec![n, b, 9216 - b]
    } else {
        vec![n, 0, b]
    };
    Ok(inner)
}
pub(crate) fn limit(r: &Request) -> Result<usize> {
    let (n, _, _, _) = metadata(r)?;
    Ok(if complete(r) {
        n * (3 * C + KV + C / 256 + if q4(r) { 1088 } else { 0 })
    } else {
        crate::mlp_stream::limit(&mlp(r)?)? + if compact(r) {0}else{n * KV}
    })
}
pub struct PreparedAttentionMlp {
    request: Request,
    stream: Option<crate::PreparedMlpStream>,
    values: Vec<f32>,
    quant: Option<QuantizedRows>,
    gated: Vec<f32>,
    ax: Vec<f32>,
}
impl PreparedAttentionMlp {
    pub(crate) fn decode(r: &Request, p: &[u8]) -> Result<Self> {
        let (n, _, offset, _) = metadata(r)?;
        if complete(r) {
            let (stream, prefix) = if q4(r) {
                if p.first() != Some(&4) || p.len() < 5 {
                    return Err("attention MLP Q4 direction".into());
                }
                let len = u32::from_le_bytes(p[1..5].try_into().unwrap()) as usize;
                if len.checked_add(5 + 2 * offset * KV) != Some(p.len()) {
                    return Err("attention MLP Q4 prefix length".into());
                }
                (&p[5..5 + len], unpack_bf(&p[5 + len..])?)
            } else {
                (p, vec![])
            };
            return Ok(Self {
                request: r.clone(),
                stream: Some(crate::PreparedMlpStream::decode(&mlp(r)?, stream)?),
                values: prefix,
                quant: None,
                gated: vec![],
                ax: vec![],
            });
        }
        let base = n * (if compact(r) { C }else{2*C} + KV);
        let gate = if q4(r) { n * 1024 } else { 0 };
        let bf = base + gate + offset * KV;
        let end = 1 + 2 * bf;
        let scales = n * C / 256;
        let a = if q4(r) { n * 64 } else { 0 };
        if p.first() != Some(&if compact(r) {6}else if q4(r) { 5 } else { 2 })
            || p.len() != end + n * C + 4 * (scales + a)
        {
            return Err("attention MLP carry length/direction".into());
        }
        // Unpack directly into the final layout; do not split off and copy
        // the unchanged prefix twice on every legacy or Q4 request.
        let mut v = vec![0.; base + offset * KV];
        crate::bf16_codec::unpack(&p[1..1+2*base], &mut v[..base]);
        crate::bf16_codec::unpack(&p[1+2*(base+gate)..end], &mut v[base..]);
        if !v.iter().all(|v| v.is_finite()) {return Err("attention MLP carry finite".into());}
        let gated = unpack_bf(&p[1+2*base..1+2*(base+gate)])?;
        let sx = unpack_f32(&p[end + n * C..end + n * C + 4 * scales])?;
        let q = QuantizedRows::from_bytes(n, C, &p[end..end + n * C], &sx)?;
        let ax = unpack_f32(&p[end + n * C + 4 * scales..])?;
        Ok(Self {
            request: r.clone(),
            stream: None,
            values: v,
            quant: Some(q),
            gated,
            ax,
        })
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
            return Err("attention MLP identity".into());
        }
        let (n, _, p, layer) = metadata(r)?;
        let count = n * C;
        if let Some(stream) = self.stream {
            let (mut both, mut used) = crate::profile::measure("bridge_mlp_complete", || {
                stream.evaluate(&mlp(r)?, m, read)
            })?;
            if both.len() != 2 * count {
                return Err("attention MLP complete shape".into());
            }
            let q = crate::profile::measure("bridge_attention_quantize_once", || {
                crate::int8_kernel::quantize_rows(&both[count..], n, C)
            })?;
            let mut kr = r.clone();
            kr.encoding = "bf16-block256-exact-v1".into();
            kr.op = "attention_kv_integer".into();
            kr.tensor = format!(
                "model.language_model.layers.{}.self_attn.k_proj.weight",
                layer + 1
            );
            kr.dims = vec![n, p];
            kr.scalars.clear();
            kr.aux.clear();
            let (kv, bytes) = crate::profile::measure("bridge_attention_kv", || {
                crate::attention_fusion::evaluate_shared(&kr, &both[count..], m, read, Some(&q))
            })?;
            used += bytes;
            let mut gated = vec![];
            let mut ax = vec![];
            if q4(r) {
                let mut qr = r.clone();
                qr.encoding = "bf16-block256-exact-v1".into();
                qr.op = "attention_q_gqa_integer".into();
                qr.tensor = format!(
                    "model.language_model.layers.{}.self_attn.q_proj.weight",
                    layer + 1
                );
                qr.dims = vec![n, n + p, p, 0, 4];
                qr.scalars.clear();
                qr.aux.clear();
                let (a, bytes) = crate::profile::measure("bridge_q_A_once", || {
                    crate::attention_fusion::prepare_q_a(&qr, &both[count..], m, read)
                })?;
                used += bytes;
                ax = a;
                let payload = q_payload(&both[count..], &kv, &self.values, n, p, 0, 1);
                let (g, bytes) = crate::profile::measure("bridge_q_first4", || {
                    crate::attention_fusion::evaluate_shared_with_a(
                        &qr,
                        &payload,
                        m,
                        read,
                        Some(&q),
                        Some(&ax),
                    )
                })?;
                used += bytes;
                gated = g;
            }
            both.extend(kv);
            q.append_wire_values(&mut both);
            both.extend_from_slice(&q.scales()[..n * C / 256]);
            both.extend(gated);
            both.extend(ax);
            return Ok((both, used));
        }
        let v = self.values;
        let q = self.quant.ok_or("attention MLP quant state")?;
        let total = n + p;
        let kv_start=if compact(r) {count}else{2*count};
        let kv = &v[kv_start..kv_start + n * KV];
        let prefix = &v[kv_start + n * KV..];
        let first = if q4(r) { 4 } else { 0 };
        let heads = 16 - first;
        let norm=if compact(r) {&[][..]}else{&v[count..2*count]};
        let payload = q_payload(norm, kv, prefix, n, p, first / 4, heads / 4);
        let root = format!("model.language_model.layers.{layer}.self_attn");
        let mut qr = r.clone();
        qr.op = "attention_q_gqa_integer".into();
        qr.encoding = "bf16-block256-exact-v1".into();
        qr.tensor = format!("{root}.q_proj.weight");
        qr.dims = vec![n, total, p, first, heads];
        qr.scalars.clear();
        qr.aux.clear();
        let (g, mut used) = crate::profile::measure("bridge_attention_q_gqa", || {
            crate::attention_fusion::evaluate_shared_with_a(
                &qr,
                &payload,
                m,
                read,
                Some(&q),
                if q4(r) { Some(&self.ax) } else { None },
            )
        })?;
        let gated = if q4(r) {
            let mut all = Vec::with_capacity(n * 4096);
            for t in 0..n {
                all.extend_from_slice(&self.gated[t * 1024..(t + 1) * 1024]);
                all.extend_from_slice(&g[t * 3072..(t + 1) * 3072]);
            }
            all
        } else {
            g
        };
        let mut out = qr;
        out.op = "lora_integer".into();
        out.tensor = format!("{root}.o_proj.weight");
        out.dims = vec![n, C, 4096, 0];
        out.scalars = vec![2.];
        out.aux = vec![
            format!("{root}.o_proj.lora_A.weight"),
            format!("{root}.o_proj.lora_B.weight"),
        ];
        let (attention, bytes) = crate::profile::measure("bridge_attention_out", || {
            crate::evaluate_integer_with_ax(&out, &gated, m, read, None, None)
        })?;
        used += bytes;
        let mut pair = v[..count].to_vec();
        pair.extend(attention);
        let (mut state, bytes) = crate::profile::measure("bridge_mlp_front", || {
            crate::mlp_stream::prepare_direct(&mlp(r)?, pair, m, read)
        })?;
        used += bytes;
        if !compact(r) {state.extend_from_slice(kv);}
        Ok((state, used))
    }
}
pub(crate) fn append(b: &mut Vec<u8>, r: &Request, v: &[f32]) -> Result<()> {
    let (n, _, _, _) = metadata(r)?;
    if v.len() != limit(r)? || !v.iter().all(|v| v.is_finite()) {
        return Err("attention MLP reply count/finite".into());
    }
    if complete(r) {
        let count = n * (2 * C + KV);
        let qend = count + n * C;
        let send = qend + n * C / 256;
        let gend = send + if q4(r) { n * 1024 } else { 0 };
        b.push(if q4(r) { 4 } else { 1 });
        append_bf(b, &v[..count])?;
        if q4(r) {
            append_bf(b, &v[send..gend])?;
        }
        let q = &v[count..qend];
        if q.iter()
            .any(|v| *v < -127. || *v > 127. || *v != v.trunc() || v.to_bits() == 0x80000000)
            || v[qend..send].iter().any(|v| *v <= 0.)
        {
            return Err("attention MLP integer/scale reply".into());
        }
        b.extend(q.iter().map(|v| *v as i8 as u8));
        for v in &v[qend..send] {
            b.extend(v.to_le_bytes());
        }
        for v in &v[gend..] {
            b.extend(v.to_le_bytes());
        }
        return Ok(());
    }
    if compact(r) {b.push(7);return crate::mlp_stream::append(b,&mlp(r)?,v);}
    b.push(3);
    let end = v.len() - n * KV;
    crate::mlp_stream::append(b, &mlp(r)?, &v[..end])?;
    if !crate::bf16_codec::all_bf16(&v[end..]) {
        return Err("attention MLP KV precision".into());
    }
    let pos = b.len();
    b.resize(pos + 2 * n * KV, 0);
    crate::bf16_codec::pack(&v[end..], &mut b[pos..]);
    Ok(())
}
pub(crate) fn decode_reply(r: &Request, p: &[u8]) -> Result<Vec<f32>> {
    let (n, _, _, _) = metadata(r)?;
    if complete(r) {
        let count = n * (2 * C + KV);
        let gate = if q4(r) { n * 1024 } else { 0 };
        let a = if q4(r) { n * 64 } else { 0 };
        let end = 1 + (count + gate) * 2;
        let scales = n * C / 256;
        if p.first() != Some(&if q4(r) { 4 } else { 1 })
            || p.len() != end + n * C + 4 * (scales + a)
        {
            return Err("attention MLP reply length".into());
        }
        let mut bf = unpack_bf(&p[1..end])?;
        let gated = bf.split_off(count);
        let sx = unpack_f32(&p[end + n * C..end + n * C + 4 * scales])?;
        let ax = unpack_f32(&p[end + n * C + 4 * scales..])?;
        if p[end..end + n * C].contains(&128) || sx.iter().any(|v| *v <= 0.) {
            return Err("attention MLP reply lanes".into());
        }
        bf.extend(p[end..end + n * C].iter().map(|b| *b as i8 as f32));
        bf.extend(sx);
        bf.extend(gated);
        bf.extend(ax);
        return Ok(bf);
    }
    if compact(r) {
        if p.first()!=Some(&7) {return Err("compact attention MLP reply direction".into());}
        let v=crate::mlp_stream::decode_values(&mlp(r)?,&p[1..])?;
        if v.len()!=limit(r)? {return Err("compact attention MLP reply count".into());}return Ok(v);
    }
    if p.first() != Some(&3) || p.len() < 1 + 2 * n * KV {
        return Err("attention MLP reply direction".into());
    }
    let end = p.len() - 2 * n * KV;
    let mut v = crate::mlp_stream::decode_values(&mlp(r)?, &p[1..end])?;
    let mut kv = vec![0.; n * KV];
    crate::bf16_codec::unpack(&p[end..], &mut kv);
    v.extend(kv);
    if v.len() != limit(r)? || !v.iter().all(|v| v.is_finite()) {
        return Err("attention MLP reply finite/count".into());
    }
    Ok(v)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn req() -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"attention_finish_mlp_front","encoding":NAME,"tensor":"model.language_model.layers.3.post_attention_layernorm.weight","dims":[1,4608,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.4.input_layernorm.weight"]})).unwrap()
    }
    #[test]
    fn compact_carry_omits_norm_and_checks_direction() {
        let mut r=req();r.op="attention_finish_mlp_front_q4_compact".into();
        let bf=C+KV+1024+45*KV;
        let mut p=vec![6];p.extend(vec![0;2*bf+C]);
        for _ in 0..C/256 {p.extend(1f32.to_le_bytes());}
        for _ in 0..64 {p.extend(0.01234567f32.to_le_bytes());}
        let state=PreparedAttentionMlp::decode(&r,&p).unwrap();
        assert_eq!(state.values.len(),C+KV+45*KV);
        assert_eq!(state.gated.len(),1024);assert_eq!(state.ax.len(),64);
        assert_eq!(limit(&r).unwrap(),crate::mlp_stream::limit(&mlp(&r).unwrap()).unwrap());
        p[0]=5;assert!(PreparedAttentionMlp::decode(&r,&p).is_err());
        p[0]=6;p.pop();assert!(PreparedAttentionMlp::decode(&r,&p).is_err());
    }
    #[test]
    fn checked_carry_identity_and_bounds() {
        let r = req();
        let count = 2 * C + KV + 45 * KV;
        let mut p = vec![2];
        p.extend(vec![0; 2 * count + C]);
        for _ in 0..C / 256 {
            p.extend(1f32.to_le_bytes());
        }
        let state = PreparedAttentionMlp::decode(&r, &p).unwrap();
        let mut changed = r.clone();
        changed.dims.clear();
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![],
        };
        assert_eq!(
            state
                .evaluate(&changed, &m, &mut |_, _| -> Result<Vec<u8>> {
                    panic!("identity read")
                })
                .unwrap_err(),
            "attention MLP identity"
        );
        for dims in [
            vec![],
            vec![0, 4608, 45],
            vec![90, 4608, 45],
            vec![1, 9216, 45],
            vec![1, 4609, 45],
            vec![1, 4608, 133],
            vec![usize::MAX, 4608, 45],
        ] {
            let mut bad = r.clone();
            bad.dims = dims;
            assert!(metadata(&bad).is_err());
        }
        p.push(0);
        assert!(PreparedAttentionMlp::decode(&r, &p).is_err());
        p.pop();
        p[1 + 2 * count] = 128;
        assert!(PreparedAttentionMlp::decode(&r, &p).is_err());
        p[1 + 2 * count] = 0;
        p[1..3].copy_from_slice(&0x7fc0u16.to_le_bytes());
        assert!(PreparedAttentionMlp::decode(&r, &p).is_err());
    }
    #[test]
    fn reply_preserves_integer_scales_and_negative_zero() {
        let r = Request {
            op: "mlp_complete_attention_kv".into(),
            tensor: "model.language_model.layers.2.post_attention_layernorm.weight".into(),
            aux: vec!["model.language_model.layers.3.input_layernorm.weight".into()],
            dims: vec![1, 4608, 4608, 45],
            ..req()
        };
        let mut v = vec![-0.; 2 * C + KV];
        v.extend(vec![-127.; C]);
        v.extend(vec![0.0123; C / 256]);
        let frame = crate::encode(&r, &v).unwrap();
        assert_eq!(
            crate::decode(&frame)
                .unwrap()
                .1
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            v.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
#[cfg(test)]
mod q4_tests {
    use super::*;
    #[test]
    fn q4_carry_preserves_unrounded_a_and_checks_direction() {
        let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_complete_attention_kv_q4","encoding":NAME,"tensor":"model.language_model.layers.2.post_attention_layernorm.weight","dims":[1,1280,7936,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.3.input_layernorm.weight"]})).unwrap();
        let mut state = vec![-0.; 2 * C + KV];
        state.extend(vec![-127.; C]);
        state.extend(vec![0.0123; C / 256]);
        state.extend(vec![-0.; 1024]);
        state.extend(vec![0.01234567; 64]);
        let frame = crate::encode(&r, &state).unwrap();
        let (_, decoded) = crate::decode(&frame).unwrap();
        assert_eq!(
            state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let mut stream = vec![];
        crate::mlp_stream::append(
            &mut stream,
            &mlp(&r).unwrap(),
            &vec![1.; 2 * C + 1280 + C / 256 + 1280 / 256 + 3 * 64],
        )
        .unwrap();
        let mut first = vec![4];
        first.extend((stream.len() as u32).to_le_bytes());
        first.extend(stream);
        first.extend(vec![0; 45 * KV * 2]);
        assert!(PreparedAttentionMlp::decode(&r, &first).is_ok());
        first[0] = 1;
        assert!(PreparedAttentionMlp::decode(&r, &first).is_err());
        r.op = "attention_finish_mlp_front_q4".into();
        r.tensor = "model.language_model.layers.3.post_attention_layernorm.weight".into();
        r.aux = vec!["model.language_model.layers.4.input_layernorm.weight".into()];
        r.dims = vec![1, 5120, 45];
        let mut wire = vec![5];
        append_bf(&mut wire, &state[..2 * C + KV]).unwrap();
        append_bf(
            &mut wire,
            &state[3 * C + KV + C / 256..3 * C + KV + C / 256 + 1024],
        )
        .unwrap();
        wire.extend(vec![0; 45 * KV * 2]);
        wire.extend(vec![129; C]);
        for _ in 0..C / 256 {
            wire.extend(0.0123f32.to_le_bytes());
        }
        for _ in 0..64 {
            wire.extend(0.01234567f32.to_le_bytes());
        }
        let parsed = PreparedAttentionMlp::decode(&r, &wire).unwrap();
        assert_eq!(parsed.ax[0].to_bits(), 0.01234567f32.to_bits());
        assert_eq!(parsed.gated[0].to_bits(), (-0f32).to_bits());
        wire[0] = 2;
        assert!(PreparedAttentionMlp::decode(&r, &wire).is_err());
        wire[0] = 5;
        let len = wire.len();
        wire[len - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(PreparedAttentionMlp::decode(&r, &wire).is_err());
    }
}
