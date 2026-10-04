//! Client-held MLP generation chunks. Input q/A and product q are computed once.
use crate::{
    int8_kernel::QuantizedRows,
    prepared_weights::{LoadedWeight, WeightBuffer},
    Manifest, Request, Result,
};
pub(crate) const NAME: &str = "mlp-stream-exact-v1";
const C: usize = 2560;
const H: usize = 9216;
const R: usize = 64;
pub(crate) const STEP: usize = if cfg!(feature = "experimental-mlp-half") { 128 } else { 256 };
fn shape(r: &Request) -> Result<(usize, usize, usize)> {
    if r.encoding != NAME
        || !matches!(
            r.op.as_str(),
            "mlp_stream_prepare" | "mlp_stream_next" | "mlp_stream_finish" | "mlp_stream_complete"
        )
        || r.dims.len() != 3
        || r.aux.len() != 1
        || r.scalars.len() != 2
        || r.scalars[0].to_bits() != 2f32.to_bits()
        || r.scalars[1].to_bits() != 1e-6f32.to_bits()
    {
        return Err("MLP stream metadata".into());
    }
    let (n, begin, count) = (r.dims[0], r.dims[1], r.dims[2]);
    if n == 0
        || n > 89
        || begin > H
        || begin % STEP != 0
        || count % STEP != 0
        || begin.checked_add(count).is_none_or(|end| end > H)
        || (r.op == "mlp_stream_complete" && begin + count != H)
        || if r.op == "mlp_stream_finish" {
            begin != H || count != 0
        } else {
            count == 0 || (r.op == "mlp_stream_prepare") != (begin == 0)
        }
    {
        return Err("MLP stream bounds".into());
    }
    root(r)?;
    Ok((n, begin, count))
}
fn root(r: &Request) -> Result<String> {
    let p = r
        .tensor
        .strip_suffix(".post_attention_layernorm.weight")
        .ok_or("MLP stream norm")?;
    let s = p
        .strip_prefix("model.language_model.layers.")
        .ok_or("MLP stream layer")?;
    let i: usize = s.parse().map_err(|_| "MLP stream layer")?;
    if i >= 31
        || s != i.to_string()
        || r.aux[0]
            != format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                i + 1
            )
    {
        return Err("MLP stream next norm".into());
    }
    Ok(p.into())
}
fn length(n: usize, done: usize) -> usize {
    n * (2 * C + done + C / 256 + done / 256 + 3 * R)
}
pub(crate) fn limit(r: &Request) -> Result<usize> {
    let (n, b, c) = shape(r)?;
    Ok(length(n, b + c).max(2 * n * C))
}
struct Parts {
    residual: Vec<f32>,
    q: QuantizedRows,
    ax: Vec<f32>,
    product: Vec<u8>,
    // Original BF16 values for an incomplete block; never quantized alone.
    pending: Vec<f32>,
    scales: Vec<f32>,
    down: Vec<f32>,
    done: usize,
}
impl Parts {
    fn flat(self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(length(n, self.done));
        out.extend(self.residual);
        self.q.append_wire_values(&mut out);
        out.extend_from_slice(&self.q.scales()[..n * C / 256]);
        out.extend(self.ax);
        out.extend(self.product.into_iter().map(|b| b as i8 as f32));
        out.extend(self.pending);
        out.extend(self.scales);
        out.extend(self.down);
        out
    }
}
/// Opaque finite, shape-checked state remains bound to the complete request.
/// ```compile_fail
/// let mut state: imajev_runtime::PreparedMlpStream = todo!();
/// state.request.step += 1;
/// ```
pub struct PreparedMlpStream {
    request: Request,
    parts: Option<Parts>,
    plain: Vec<f32>,
}
pub(crate) fn prepare_direct<F,B>(r:&Request,plain:Vec<f32>,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    let(n,_,_)=shape(r)?;
    if r.op!="mlp_stream_prepare" || plain.len()!=2*n*C || !plain.iter().all(|v|v.is_finite()) || !crate::bf16_codec::all_bf16(&plain) {return Err("MLP direct preparation input".into());}
    PreparedMlpStream{request:r.clone(),parts:None,plain}.evaluate(r,m,read)
}
fn f32s(p: &[u8]) -> Result<Vec<f32>> {
    let v: Vec<_> = p
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if p.len() % 4 != 0 || !v.iter().all(|v| v.is_finite()) {
        return Err("MLP stream finite floats".into());
    }
    Ok(v)
}
impl PreparedMlpStream {
    pub(crate) fn decode(r: &Request, p: &[u8]) -> Result<Self> {
        let (n, begin, _) = shape(r)?;
        if r.op == "mlp_stream_prepare" {
            if p.first() != Some(&0) {
                return Err("MLP stream prepare direction".into());
            }
            let x = crate::block_codec::decode(&p[1..])?;
            if x.len() != 2 * n * C || !crate::bf16_codec::all_bf16(&x) {
                return Err("MLP stream initial precision".into());
            }
            return Ok(Self {
                request: r.clone(),
                parts: None,
                plain: x,
            });
        }
        let (done, parts) = read_parts(r, p)?;
        if done != begin {
            return Err("MLP stream progress".into());
        }
        Ok(Self {
            request: r.clone(),
            parts: Some(parts),
            plain: vec![],
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
            return Err("MLP stream identity".into());
        }
        let (n, begin, count) = shape(r)?;
        let p = root(r)?;
        let mut inner = r.clone();
        inner.op = "mlp_prepare_down".into();
        inner.encoding = crate::mlp_pipeline::NAME.into();
        inner.dims = vec![n, C];
        crate::mlp_pipeline::validate_scoped(&inner, m, false)?;
        let mut bytes = 0;
        let mut parts = if let Some(parts) = self.parts {
            parts
        } else {
            let mut nr = inner.clone();
            nr.op = "add_norm_bf16".into();
            nr.dims = vec![n, C];
            nr.scalars = vec![1e-6];
            nr.aux.clear();
            let t = m.tensors.iter().find(|t| t.name == r.tensor).unwrap();
            let (w, used) = crate::load_prepared_weight(t, &nr, &mut *read)?;
            bytes += used;
            let mut both = crate::execute(&nr, &self.plain, &w)?;
            let norm = &both[n * C..];
            let q = crate::profile::measure("stream_input_quantize_once", || {
                crate::int8_kernel::quantize_rows(norm, n, C)
            })?;
            let mut ax = vec![];
            for name in ["gate_proj", "up_proj"] {
                let t = m
                    .tensors
                    .iter()
                    .find(|t| t.name == format!("{p}.mlp.{name}.lora_A.weight"))
                    .unwrap();
                let mut ar = inner.clone();
                ar.op = "matmul".into();
                ar.dims = vec![n, R, C];
                ar.aux.clear();
                ar.scalars.clear();
                let (w, used) = crate::load_prepared_weight(t, &ar, &mut *read)?;
                bytes += used;
                ax.extend(crate::profile::measure("stream_input_A_once", || {
                    crate::matrix_loaded(norm, &w, n, R, C)
                })?);
            }
            both.truncate(n * C);
            Parts {
                residual: both,
                q,
                ax,
                product: vec![],
                pending: vec![],
                scales: vec![],
                down: vec![0.; n * R],
                done: 0,
            }
        };
        if r.op == "mlp_stream_finish" {
            return finish(parts, r, inner, &p, n, m, read);
        }
        let mut gr = inner.clone();
        gr.op = "mlp_gate_up_reuse".into();
        gr.encoding = crate::projection_codec::NAME.into();
        gr.tensor = format!("{p}.mlp.gate_proj.weight");
        gr.aux = vec![format!("{p}.mlp.up_proj.weight")];
        gr.dims = vec![n, count, C, begin, R];
        gr.scalars = vec![2.];
        let (product, used) =
            crate::mlp_reuse::evaluate_prepared(&gr, &parts.q, &parts.ax, m, read)?;
        bytes += used;
        let t = m
            .tensors
            .iter()
            .find(|t| t.name == format!("{p}.mlp.down_proj.lora_A.weight"))
            .unwrap();
        let mut ar = inner.clone();
        ar.op = "matmul".into();
        ar.dims = vec![n, R, H];
        ar.aux.clear();
        ar.scalars.clear();
        let (w, used) = crate::load_prepared_weight(t, &ar, &mut *read)?;
        bytes += used;
        let LoadedWeight::Prepared(w) = w else {
            return Err("MLP stream fixed F32 A required".into());
        };
        parts.down = crate::profile::measure("stream_down_A_continue", || {
            crate::f32_output::continue_columns(&product, &parts.down, &w, n, R, H, begin, count)
        })?;
        crate::profile::measure("stream_product_quantize_once", || {
            extend_product(&mut parts, &product, n, count)
        })?;
        if r.op == "mlp_stream_complete" {
            let (out, used) = finish(parts, r, inner, &p, n, m, read)?;
            return Ok((out, bytes + used));
        }
        Ok((parts.flat(n), bytes))
    }
}
/// Preserve block256 scales regardless of a query's 128-row boundary.
fn extend_product(parts: &mut Parts, product: &[f32], n: usize, count: usize) -> Result<()> {
    let begin = parts.done;
    let old_full = begin / 256 * 256;
    let old_rem = begin % 256;
    let available = old_rem + count;
    let full = available / 256 * 256;
    let rem = available % 256;
    if product.len() != n * count || parts.pending.len() != n * old_rem {
        return Err("MLP stream product precision/shape".into());
    }
    // Old aligned requests avoid copying their BF16 chunk into another buffer.
    let mut combined = Vec::new();
    let mut pending = Vec::with_capacity(n * rem);
    if old_rem != 0 || rem != 0 {
        combined.reserve(n * full);
        for token in 0..n {
            let old = &parts.pending[token * old_rem..(token + 1) * old_rem];
            let fresh = &product[token * count..(token + 1) * count];
            if full == 0 {
                pending.extend_from_slice(old);
                pending.extend_from_slice(fresh);
            } else {
                combined.extend_from_slice(old);
                combined.extend_from_slice(&fresh[..full - old_rem]);
                pending.extend_from_slice(&fresh[full - old_rem..]);
            }
        }
    }
    let quantized = if full == 0 { None } else {
        Some(crate::int8_kernel::quantize_rows(
            if old_rem == 0 && rem == 0 { product } else { &combined }, n, full)?)
    };
    let mut merged = Vec::with_capacity(n * (old_full + full));
    let mut scales = Vec::with_capacity(n * ((old_full + full) / 256));
    for token in 0..n {
        merged.extend_from_slice(&parts.product[token * old_full..(token + 1) * old_full]);
        scales.extend_from_slice(&parts.scales[token * (old_full / 256)..(token + 1) * (old_full / 256)]);
        if let Some(chunk) = &quantized {
            merged.extend(chunk.values()[token * full..(token + 1) * full].iter().map(|v| *v as i8 as u8));
            scales.extend_from_slice(&chunk.scales()[token * (full / 256)..(token + 1) * (full / 256)]);
        }
    }
    parts.product = merged;
    parts.scales = scales;
    parts.pending = pending;
    parts.done = begin + count;
    Ok(())
}
fn finish<F, B>(
    parts: Parts,
    r: &Request,
    inner: Request,
    p: &str,
    n: usize,
    m: &Manifest,
    read: &mut F,
) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: WeightBuffer,
{
    debug_assert_eq!(parts.done, H);
    let mut bytes = 0;
    let q = QuantizedRows::from_bytes(n, H, &parts.product, &parts.scales)?;
    let mut dr = inner.clone();
    dr.op = "lora_integer_reuse".into();
    dr.encoding = crate::projection_codec::NAME.into();
    dr.tensor = format!("{p}.mlp.down_proj.weight");
    dr.dims = vec![n, C, H, 0, R];
    dr.aux = vec![
        format!("{p}.mlp.down_proj.lora_A.weight"),
        format!("{p}.mlp.down_proj.lora_B.weight"),
    ];
    dr.scalars = vec![2.];
    let (y, used) = crate::projection_reuse::evaluate_prepared(&dr, &q, &parts.down, m, read)?;
    bytes += used;
    let mut pair = parts.residual;
    pair.extend(y);
    let mut nr = inner;
    nr.op = "add_norm_bf16".into();
    nr.tensor = r.aux[0].clone();
    nr.aux.clear();
    nr.dims = vec![n, C];
    nr.scalars = vec![1e-6];
    let t = m.tensors.iter().find(|t| t.name == nr.tensor).unwrap();
    let (w, used) = crate::load_prepared_weight(t, &nr, &mut *read)?;
    bytes += used;
    Ok((crate::execute(&nr, &pair, &w)?, bytes))
}
fn read_parts(r: &Request, p: &[u8]) -> Result<(usize, Parts)> {
    let (n, begin, count) = shape(r)?;
    if p.len() < 5 || !matches!(p[0], 1 | 2) {
        return Err("MLP stream state direction".into());
    }
    let done = u32::from_le_bytes(p[1..5].try_into().unwrap()) as usize;
    if done == 0
        || done > H
        || done % STEP != 0
        || !(done == begin || done == begin + count)
        || p[0] != if done % 256 == 0 {1} else {2}
        || p.len() != 5 + n * (C * 2 + C + done + done % 256 + (C / 256 + done / 256 + 3 * R) * 4)
    {
        return Err("MLP stream state size/progress".into());
    }
    let mut cursor = 5;
    let mut residual = vec![0.; n * C];
    crate::bf16_codec::unpack(&p[cursor..cursor + n * C * 2], &mut residual);
    cursor += n * C * 2;
    if !residual.iter().all(|v| v.is_finite()) {
        return Err("MLP stream residual finite".into());
    }
    let integers = &p[cursor..cursor + n * C];
    cursor += n * C;
    let sx = f32s(&p[cursor..cursor + n * C / 256 * 4])?;
    cursor += n * C / 256 * 4;
    let q = QuantizedRows::from_bytes(n, C, integers, &sx)?;
    let ax = f32s(&p[cursor..cursor + n * 2 * R * 4])?;
    cursor += n * 2 * R * 4;
    let full = done / 256 * 256;
    let product = p[cursor..cursor + n * full].to_vec();
    cursor += n * full;
    let mut pending = vec![0.; n * (done % 256)];
    crate::bf16_codec::unpack(&p[cursor..cursor + pending.len() * 2], &mut pending);
    cursor += pending.len() * 2;
    if !pending.iter().all(|v|v.is_finite()) {return Err("MLP stream pending finite".into());}
    if product.contains(&128) {
        return Err("MLP stream product integer".into());
    }
    let scales = f32s(&p[cursor..cursor + n * (done / 256) * 4])?;
    cursor += n * (done / 256) * 4;
    if scales.iter().any(|v| *v <= 0.) {
        return Err("MLP stream product scale".into());
    }
    let down = f32s(&p[cursor..])?;
    Ok((
        done,
        Parts {
            residual,
            q,
            ax,
            product,
            pending,
            scales,
            down,
            done,
        },
    ))
}
pub(crate) fn append(b: &mut Vec<u8>, r: &Request, x: &[f32]) -> Result<()> {
    let (n, begin, count) = shape(r)?;
    if x.len() == 2 * n * C {
        if !crate::bf16_codec::all_bf16(x) {
            return Err("MLP stream plain precision".into());
        }
        b.push(0);
        return crate::block_codec::append(b, x);
    }
    let done = if x.len() == length(n, begin + count) {
        begin + count
    } else if x.len() == length(n, begin) {
        begin
    } else {
        return Err("MLP stream flat length".into());
    };
    if done == 0 || !crate::bf16_codec::all_bf16(&x[..n * C]) || !x.iter().all(|v| v.is_finite()) {
        return Err("MLP stream state precision".into());
    }
    let mut p = vec![if done % 256 == 0 {1} else {2}];
    p.extend_from_slice(&(done as u32).to_le_bytes());
    let start = p.len();
    p.resize(start + n * C * 2, 0);
    crate::bf16_codec::pack(&x[..n * C], &mut p[start..]);
    let mut cursor = n * C;
    let start = p.len();
    p.resize(start + n * C, 0);
    crate::projection_codec::pack_integers(&x[cursor..cursor + n * C], &mut p[start..])?;
    cursor += n * C;
    let sx = n * C / 256;
    let tail = sx + 2 * n * R;
    for v in &x[cursor..cursor + tail] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    cursor += tail;
    let start = p.len();
    let full = done / 256 * 256;
    p.resize(start + n * full, 0);
    crate::projection_codec::pack_integers(&x[cursor..cursor + n * full], &mut p[start..])?;
    cursor += n * full;
    let pending = n * (done % 256);
    if !crate::bf16_codec::all_bf16(&x[cursor..cursor + pending]) {return Err("MLP stream pending precision".into());}
    let start = p.len();
    p.resize(start + 2 * pending, 0);
    crate::bf16_codec::pack(&x[cursor..cursor + pending], &mut p[start..]);
    cursor += pending;
    for v in &x[cursor..] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    // Finite/BF16/length/integer checks above already validate each lane.
    // Check the two positive scale spans directly instead of decoding the
    // entire freshly encoded carry into another owned Parts allocation.
    let input_scales = n * C * 2;
    let product_scales = n * (C * 2 + C / 256 + 2 * R + done);
    if x[input_scales..input_scales + n * C / 256]
        .iter()
        .any(|v| *v <= 0.)
        || x[product_scales..product_scales + n * (done / 256)]
            .iter()
            .any(|v| *v <= 0.)
    {
        return Err("MLP stream scale".into());
    }
    if b.len() + p.len() + 32 > 2_000_000 {
        return Err("MLP stream frame size".into());
    }
    b.extend(p);
    Ok(())
}
pub(crate) fn decode_values(r: &Request, p: &[u8]) -> Result<Vec<f32>> {
    let (n, _, _) = shape(r)?;
    if p.first() == Some(&0) {
        let x = crate::block_codec::decode(&p[1..])?;
        if x.len() != 2 * n * C || !crate::bf16_codec::all_bf16(&x) {
            return Err("MLP stream plain length/precision".into());
        }
        return Ok(x);
    }
    Ok(read_parts(r, p)?.1.flat(n))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_stream_next","tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[87,4096,5120],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"],"encoding":NAME})).unwrap()
    }
    #[cfg(feature = "experimental-mlp-half")]
    #[test]
    fn half_chunks_keep_original_block256_quantization_and_wire_bits() {
        for n in [1,7,87,89] {
            let input: Vec<f32> = (0..n*H).map(|i| {
                let x = ((i*71 % 1023) as f32 - 511.) / 32.;
                f32::from_bits(x.to_bits() & 0xffff0000)
            }).collect();
            let expected = crate::int8_kernel::quantize_rows(&input,n,H).unwrap();
            for chunks in [vec![128,128,H-256],vec![6016,3200],vec![128,256,128,H-512]] {
                let mut parts = Parts { residual:vec![-0.;n*C],
                    q:QuantizedRows::from_bytes(n,C,&vec![0;n*C],&vec![1.;n*C/256]).unwrap(),
                    ax:vec![0.1234567;n*2*R],product:vec![],pending:vec![],scales:vec![],
                    down:vec![-0.1234567;n*R],done:0 };
                let mut begin=0;
                for count in chunks {
                    let mut fresh=Vec::with_capacity(n*count);
                    for t in 0..n {fresh.extend_from_slice(&input[t*H+begin..t*H+begin+count]);}
                    extend_product(&mut parts,&fresh,n,count).unwrap();
                    begin+=count;
                    if begin%256==128 {
                        for t in 0..n {assert_eq!(parts.pending[t*128..(t+1)*128],input[t*H+begin-128..t*H+begin]);}
                    }
                    // Simulate the byte-exact client carry boundary at every chunk.
                    let mut r=request();r.dims=vec![n,begin,if begin==H{0}else{128}];
                    if begin==H {r.op="mlp_stream_finish".into();}
                    let flat=parts.flat(n);let mut encoded=Vec::new();append(&mut encoded,&r,&flat).unwrap();
                    assert_eq!(encoded[0],if begin%256==0{1}else{2});
                    let (_,decoded)=read_parts(&r,&encoded).unwrap();
                    let again=decode_values(&r,&encoded).unwrap();
                    assert!(flat.iter().zip(&again).all(|(a,b)|a.to_bits()==b.to_bits()));
                    parts=decoded;
                }
                assert!(parts.pending.is_empty());
                assert_eq!(parts.product,expected.values()[..n*H].iter().map(|v|*v as i8 as u8).collect::<Vec<_>>());
                assert!(parts.scales.iter().zip(expected.scales()).all(|(a,b)|a.to_bits()==b.to_bits()));
            }
        }
    }
    #[cfg(feature = "experimental-mlp-half")]
    #[test]
    fn half_pending_rejects_changed_tag_nonfinite_and_non_bf16() {
        let mut r=request();r.dims=vec![1,128,128];
        let parts=Parts {residual:vec![0.;C],q:QuantizedRows::from_bytes(1,C,&vec![0;C],&vec![1.;C/256]).unwrap(),
            ax:vec![0.;2*R],product:vec![],pending:vec![-0.;128],scales:vec![],down:vec![0.;R],done:128};
        let mut flat=parts.flat(1);let mut p=Vec::new();append(&mut p,&r,&flat).unwrap();
        let offset=5+3*C+4*(C/256+2*R);
        let mut bad=p.clone();bad[0]=1;assert!(read_parts(&r,&bad).is_err());
        let mut bad=p.clone();bad[offset..offset+2].copy_from_slice(&0x7f80u16.to_le_bytes());assert!(read_parts(&r,&bad).is_err());
        flat[2*C+C/256+2*R]=0.1234567;assert!(append(&mut vec![],&r,&flat).is_err());
        assert!(read_parts(&r,&p[..p.len()-1]).is_err());
    }
    #[cfg(not(feature = "experimental-mlp-half"))]
    #[test]
    fn half_boundary_requires_explicit_feature() {
        let mut r=request();r.dims=vec![1,128,128];assert!(shape(&r).is_err());
    }
    #[test]
    fn maximum_carry_is_lossless_and_bounded() {
        for n in [1, 7, 87, 89] {
            let mut r = request();
            r.dims[0] = n;
            let done = 9216;
            let mut x = vec![-0.; n * C];
            x.extend(vec![-127.; n * C]);
            x.extend(vec![0.0123; n * C / 256]);
            x.extend(vec![0.1234567; n * 2 * R]);
            x.extend(vec![127.; n * done]);
            x.extend(vec![0.009; n * (done / 256)]);
            x.extend(vec![-0.1234567; n * R]);
            let frame = crate::encode(&r, &x).unwrap();
            assert!(frame.len() < 2_000_000);
            let (_, y) = crate::decode(&frame).unwrap();
            assert!(x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(crate::decode_query(&frame).is_err());
            r.dims = [n, done, 0].to_vec();
            r.op = "mlp_stream_finish".into();
            let frame = crate::encode(&r, &x).unwrap();
            assert!(crate::decode_query(&frame).unwrap().1.values().is_none());
        }
    }
    #[test]
    fn encoder_rejects_invalid_input_and_product_scales_without_rebuilding_parts() {
        let mut r = request();
        r.dims = vec![1, 256, 256];
        let mut x = vec![0.; C * 2];
        x.extend(vec![1.; C / 256]);
        x.extend(vec![0.; R * 2]);
        x.extend(vec![0.; 256]);
        x.push(1.);
        x.extend(vec![0.; R]);
        assert!(crate::encode(&r, &x).is_ok());
        for offset in [2 * C, 2 * C + C / 256 + 2 * R + 256] {
            for value in [0., -0., -1., f32::NAN, f32::INFINITY] {
                let mut bad = x.clone();
                bad[offset] = value;
                assert!(crate::encode(&r, &bad).is_err());
            }
        }
    }
    #[test]
    fn complete_requires_a_nonempty_final_chunk_and_exact_input_progress() {
        let mut r = request();
        r.op = "mlp_stream_complete".into();
        for dims in [
            vec![1, 0, H],
            vec![1, H, 0],
            vec![1, 256, 256],
            vec![1, H - 256, 512],
        ] {
            r.dims = dims;
            assert!(shape(&r).is_err());
        }
        r.dims = vec![1, 4608, 4608];
        assert_eq!(shape(&r).unwrap(), (1, 4608, 4608));
        let parts = Parts {
            residual: vec![0.; C],
            q: QuantizedRows::from_bytes(1, C, &vec![0; C], &vec![1.; C / 256]).unwrap(),
            ax: vec![0.; 2 * R],
            product: vec![0; 4608],
            pending: vec![],
            scales: vec![1.; 4608 / 256],
            down: vec![0.; R],
            done: 4608,
        };
        let frame = crate::encode(&r, &parts.flat(1)).unwrap();
        assert!(crate::decode_query(&frame).unwrap().1.values().is_none());
        // A completed carry is a reply representation, not valid input to
        // another final-chunk query: it would repeat those columns.
        let mut finished = r.clone();
        finished.op = "mlp_stream_finish".into();
        finished.dims = vec![1, H, 0];
        let parts = Parts {
            residual: vec![0.; C],
            q: QuantizedRows::from_bytes(1, C, &vec![0; C], &vec![1.; C / 256]).unwrap(),
            ax: vec![0.; 2 * R],
            product: vec![0; H],
            pending: vec![],
            scales: vec![1.; H / 256],
            down: vec![0.; R],
            done: H,
        };
        let frame = crate::encode(&finished, &parts.flat(1)).unwrap();
        let h = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        assert!(PreparedMlpStream::decode(&r, &frame[4 + h..frame.len() - 32]).is_err());
    }
    #[test]
    fn invalid_metadata_is_rejected() {
        for dims in [
            vec![],
            vec![0, 0, 256],
            vec![90, 0, 256],
            vec![1, 1, 256],
            vec![1, 8192, 2048],
            vec![1, 256, usize::MAX],
        ] {
            let mut r = request();
            r.dims = dims;
            assert!(shape(&r).is_err());
        }
    }
    #[test]
    fn identity_and_malformed_carry_reject_before_weight_reads() {
        let mut r = request();
        r.dims = vec![1, 256, 256];
        let mut x = vec![0.; C];
        x.extend(vec![0.; C]);
        x.extend(vec![1.; C / 256]);
        x.extend(vec![0.1234567; 2 * R]);
        x.extend(vec![0.; 256]);
        x.extend(vec![1.]);
        x.extend(vec![-0.1234567; R]);
        let frame = crate::encode(&r, &x).unwrap();
        let h = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        let payload = &frame[4 + h..frame.len() - 32];
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![],
        };
        let mut bad = r.clone();
        bad.step += 1;
        assert!(PreparedMlpStream::decode(&r, payload)
            .unwrap()
            .evaluate(&bad, &m, &mut |_, _| -> Result<Vec<u8>> {
                panic!("identity read")
            })
            .is_err());
        for offset in [
            5 + C * 2,
            5 + C * 3,
            5 + C * 3 + C / 256 * 4,
            5 + C * 3 + (C / 256 + 2 * R) * 4,
        ] {
            let mut p = payload.to_vec();
            if offset == 5 + C * 2 || offset == 5 + C * 3 + (C / 256 + 2 * R) * 4 {
                p[offset] = 128;
            } else {
                p[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
            assert!(PreparedMlpStream::decode(&r, &p).is_err());
        }
        let mut p = payload.to_vec();
        p.extend([0; 4]);
        assert!(PreparedMlpStream::decode(&r, &p).is_err());
    }
}
