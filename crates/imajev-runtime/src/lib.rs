mod bf16_codec;
mod delta_simd;
pub mod int8_kernel;
pub mod profile;
mod quantize_simd;
// Qwen3.5 text primitives and dedicated decision head.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub type Result<T> = std::result::Result<T, String>;
pub const MAX_FLOATS: usize = 900_000;
pub const FUSED_WORK_LIMIT: u64 = 450_000_000;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tensor {
    pub name: String,
    pub offset: u64,
    pub rows: usize,
    pub cols: usize,
    pub dtype: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub model: String,
    pub pack_hash: String,
    pub bytes: u64,
    pub tensors: Vec<Tensor>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    pub version: u32,
    pub model: String,
    pub pack_hash: String,
    pub input_hash: String,
    pub step: u64,
    pub op: String,
    pub tensor: String,
    pub dims: Vec<usize>,
    pub scalars: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aux: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub encoding: String,
}
/// Prefix eligible for lossy activation transport. Recurrent state, gates,
/// and input token IDs remain exact F32. Input/reply lengths distinguish delta layouts.
pub fn int8_prefix(req: &Request, count: usize) -> Result<usize> {
    match req.op.as_str() {
        "embed" if req.dims.len() == 2 && count == req.dims[0] => Ok(0),
        "delta_gates" => Ok(0),
        "delta_heads_bf16" => {
            if req.dims.len() != 4 {
                return Err("INT8 delta heads shape".into());
            }
            let (n, k, v, h) = (req.dims[0], req.dims[1], req.dims[2], req.dims[3]);
            if n == 0 || n > 512 || k == 0 || k > 128 || v == 0 || v > 128 || h == 0 || h > 16 {
                return Err("INT8 delta heads shape".into());
            }
            if count == h * (n * (2 * k + v + 2) + k * v) {
                Ok(h * n * (2 * k + v))
            } else if count == h * (n * v + k * v) {
                Ok(h * n * v)
            } else {
                Err("INT8 delta heads payload".into())
            }
        }
        "delta" | "delta_bf16" => {
            if req.dims.len() != 3 {
                return Err("INT8 delta shape".into());
            }
            let (n, k, v) = (req.dims[0], req.dims[1], req.dims[2]);
            if n == 0 || n > 512 || k == 0 || k > 128 || v == 0 || v > 128 {
                return Err("INT8 delta shape".into());
            }
            if count == n * (2 * k + v + 2) + k * v {
                Ok(n * (2 * k + v))
            } else if count == n * v + k * v {
                Ok(n * v)
            } else {
                Err("INT8 delta payload".into())
            }
        }
        _ => Ok(count),
    }
}
fn wire_float_limit(r: &Request) -> usize {
    if r.op == "delta_heads_bf16" && r.encoding == "int8-block256-v1" {
        1_200_000
    } else {
        MAX_FLOATS
    }
}
/// Losslessly stores BF16-exact F32 values in two bytes and other F32 in four.
/// A bitmap marks full F32 exceptions; no rounding is performed by this codec.
pub fn encode(req: &Request, x: &[f32]) -> Result<Vec<u8>> {
    let limit = if matches!(req.encoding.as_str(), "bf16-exact" | "int8-block256-v1") {
        wire_float_limit(req)
    } else {
        450_000
    };
    if x.len() > limit || !x.iter().all(|v| v.is_finite()) {
        return Err("invalid activation".into());
    }
    let header = serde_json::to_vec(req).map_err(|e| e.to_string())?;
    if header.len() > 16_384 {
        return Err("header size".into());
    }
    let mut b = Vec::new();
    b.extend_from_slice(&(header.len() as u32).to_le_bytes());
    b.extend(header);
    match req.encoding.as_str() {
        "" => {
            for v in x {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        "bf16-exact" => {
            b.extend_from_slice(&(x.len() as u32).to_le_bytes());
            let bitmap = b.len();
            b.resize(bitmap + x.len().div_ceil(8), 0);
            if bf16_codec::all_bf16(x) {
                let begin = b.len();
                b.resize(begin + x.len() * 2, 0);
                bf16_codec::pack(x, &mut b[begin..]);
            } else {
                for (i, v) in x.iter().enumerate() {
                    let bits = v.to_bits();
                    if bits & 0xffff == 0 {
                        b.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
                    } else {
                        b[bitmap + i / 8] |= 1 << (i % 8);
                        b.extend_from_slice(&bits.to_le_bytes());
                    }
                }
            }
        }
        "int8-block256-v1" => {
            let prefix = int8_prefix(req, x.len())?;
            b.extend_from_slice(&(x.len() as u32).to_le_bytes());
            b.extend_from_slice(&(prefix as u32).to_le_bytes());
            for block in x[..prefix].chunks(256) {
                let max = block.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let scale = if max == 0. {
                    1.
                } else {
                    (max / 127.).max(f32::from_bits(1))
                };
                b.extend_from_slice(&scale.to_le_bytes());
                for &v in block {
                    b.push((v / scale).round_ties_even().clamp(-127., 127.) as i8 as u8);
                }
            }
            for v in &x[prefix..] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        _ => return Err("unsupported encoding".into()),
    }
    if b.len() + 32 > 2_000_000 {
        return Err("state size".into());
    }
    let digest = profile::measure("wire_encode_sha256", || Sha256::digest(&b));
    b.extend_from_slice(&digest);
    Ok(b)
}
pub fn decode(b: &[u8]) -> Result<(Request, Vec<f32>)> {
    if b.len() < 36 || b.len() > 2_000_000 {
        return Err("state size".into());
    }
    let n = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
    if n > 16_384 || n + 36 > b.len() {
        return Err("header/shape".into());
    }
    if profile::measure("wire_decode_sha256", || Sha256::digest(&b[..b.len() - 32])).as_slice()
        != &b[b.len() - 32..]
    {
        return Err("checksum".into());
    }
    let r: Request = serde_json::from_slice(&b[4..4 + n]).map_err(|e| e.to_string())?;
    if r.version != 1
        || r.model.len() != 64
        || r.pack_hash.len() != 64
        || r.input_hash.len() != 64
        || !r
            .model
            .bytes()
            .chain(r.pack_hash.bytes())
            .chain(r.input_hash.bytes())
            .all(|c| c.is_ascii_hexdigit())
        || r.dims.len() > 8
        || r.aux.len() > 2
        || r.aux.iter().any(|s| s.len() > 128)
        || r.scalars.len() > 8
        || !r.scalars.iter().all(|x| x.is_finite())
    {
        return Err("state identity/format".into());
    }
    let payload = &b[4 + n..b.len() - 32];
    let x: Vec<f32> = match r.encoding.as_str() {
        "" => {
            if payload.len() % 4 != 0 || payload.len() / 4 > 450_000 {
                return Err("activation shape".into());
            }
            payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        "bf16-exact" => {
            if payload.len() < 4 {
                return Err("codec count".into());
            }
            let count = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
            if count > MAX_FLOATS {
                return Err("codec count".into());
            }
            let bitmap_len = count.div_ceil(8);
            if payload.len() < 4 + bitmap_len {
                return Err("codec bitmap".into());
            }
            let bitmap = &payload[4..4 + bitmap_len];
            if count % 8 != 0 && bitmap.last().is_some_and(|v| v >> (count % 8) != 0) {
                return Err("codec padding".into());
            }
            if bitmap.iter().all(|v| *v == 0) {
                let begin = 4 + bitmap_len;
                if payload.len() != begin + count * 2 {
                    return Err("codec length".into());
                }
                let mut values = vec![0.; count];
                bf16_codec::unpack(&payload[begin..], &mut values);
                values
            } else {
                let mut cursor = 4 + bitmap_len;
                let mut values = Vec::with_capacity(count);
                for i in 0..count {
                    let full = bitmap[i / 8] & (1 << (i % 8)) != 0;
                    let width = if full { 4 } else { 2 };
                    if cursor + width > payload.len() {
                        return Err("codec truncated".into());
                    }
                    let bits = if full {
                        let bits =
                            u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
                        if bits & 0xffff == 0 {
                            return Err("codec noncanonical".into());
                        }
                        bits
                    } else {
                        (u16::from_le_bytes(payload[cursor..cursor + 2].try_into().unwrap()) as u32)
                            << 16
                    };
                    values.push(f32::from_bits(bits));
                    cursor += width;
                }
                if cursor != payload.len() {
                    return Err("codec trailing bytes".into());
                }
                values
            }
        }
        "int8-block256-v1" => {
            if payload.len() < 8 {
                return Err("INT8 codec count".into());
            }
            let count = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
            let prefix = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
            if count > wire_float_limit(&r) || prefix > count || prefix != int8_prefix(&r, count)? {
                return Err("INT8 codec bounds".into());
            }
            if payload.len() != 8 + prefix + prefix.div_ceil(256) * 4 + (count - prefix) * 4 {
                return Err("INT8 codec length".into());
            }
            let mut values = Vec::with_capacity(count);
            let mut cursor = 8;
            for start in (0..prefix).step_by(256) {
                let scale = f32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
                cursor += 4;
                if !scale.is_finite() || scale <= 0. {
                    return Err("INT8 codec scale".into());
                }
                let width = (prefix - start).min(256);
                for &byte in &payload[cursor..cursor + width] {
                    let q = byte as i8;
                    if q == i8::MIN {
                        return Err("INT8 codec range".into());
                    }
                    values.push(q as f32 * scale);
                }
                cursor += width;
            }
            for chunk in payload[cursor..].chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            values
        }
        _ => return Err("unsupported encoding".into()),
    };
    if !x.iter().all(|v| v.is_finite()) {
        return Err("invalid activation".into());
    }
    Ok((r, x))
}
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
pub fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let s = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    x.iter().zip(w).map(|(v, w)| v * s * w).collect()
}
pub fn softmax(x: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if x.len() < 2
        || !temperature.is_finite()
        || temperature <= 0.0
        || !x.iter().all(|v| v.is_finite())
    {
        return Err("invalid logits/temperature".into());
    }
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<_> = x.iter().map(|v| ((v - m) / temperature).exp()).collect();
    let s = p.iter().sum::<f32>();
    for v in &mut p {
        *v /= s;
    }
    Ok(p)
}
pub fn matrix_reference(
    x: &[f32],
    w: &[f32],
    n: usize,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if n.checked_mul(cols) != Some(x.len()) || rows.checked_mul(cols) != Some(w.len()) {
        return Err("matmul shape".into());
    }
    if n == 0 || rows == 0 || cols == 0 || n.checked_mul(rows).is_none_or(|v| v > MAX_FLOATS) {
        return Err("matmul output bounds".into());
    }
    let mut out = vec![0.0; n * rows];
    for t in 0..n {
        for r in 0..rows {
            let mut v = 0.0;
            for c in 0..cols {
                v += x[t * cols + c] * w[r * cols + c];
            }
            out[t * rows + r] = v;
        }
    }
    Ok(out)
}
/// Four independent token lanes retain the scalar accumulation order for every dot.
/// Packing is local to the operator; client-held state stays in token-major order.
pub fn matrix(x: &[f32], w: &[f32], n: usize, rows: usize, cols: usize) -> Result<Vec<f32>> {
    if n.checked_mul(cols) != Some(x.len())
        || rows.checked_mul(cols) != Some(w.len())
        || n == 0
        || rows == 0
        || cols == 0
        || n.checked_mul(rows).is_none_or(|v| v > MAX_FLOATS)
    {
        return Err("matmul shape/output bounds".into());
    }
    if n < 4 {
        return matrix_reference(x, w, n, rows, cols);
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Shape-calibrated tile sizes; larger tiles lose on short/small matrices.
        return Ok(unsafe {
            if n >= 128 && rows >= 64 {
                matrix_multi::<16>(x, w, n, rows, cols)
            } else if n >= 32 && rows >= 64 {
                matrix_multi::<8>(x, w, n, rows, cols)
            } else {
                matrix_multi::<4>(x, w, n, rows, cols)
            }
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut out = vec![0.0; n * rows];
        let mut packed = vec![0.0; cols * 4];
        for t in (0..n / 4 * 4).step_by(4) {
            for c in 0..cols {
                for lane in 0..4 {
                    packed[c * 4 + lane] = x[(t + lane) * cols + c];
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                for r in 0..rows {
                    let weights = &w[r * cols..(r + 1) * cols];
                    #[cfg(target_arch = "wasm32")]
                    let sums = unsafe { dot_four(&packed, weights) };
                    #[cfg(not(target_arch = "wasm32"))]
                    let sums = {
                        let mut sums = [0.0; 4];
                        for c in 0..cols {
                            for lane in 0..4 {
                                sums[lane] += packed[c * 4 + lane] * weights[c];
                            }
                        }
                        sums
                    };
                    for lane in 0..4 {
                        out[(t + lane) * rows + r] = sums[lane];
                    }
                }
            }
        }
        for t in n / 4 * 4..n {
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += x[t * cols + c] * w[r * cols + c];
                }
                out[t * rows + r] = sum;
            }
        }
        Ok(out)
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_four(packed: &[f32], weights: &[f32]) -> [f32; 4] {
    use core::arch::wasm32::*;
    // matrix validates lengths; each load spans exactly four adjacent F32 lanes.
    let mut sum = f32x4_splat(0.0);
    for (c, &weight) in weights.iter().enumerate() {
        let input = unsafe { v128_load(packed.as_ptr().add(c * 4).cast()) };
        sum = f32x4_add(sum, f32x4_mul(input, f32x4_splat(weight)));
    }
    let mut result = [0.0; 4];
    unsafe {
        v128_store(result.as_mut_ptr().cast(), sum);
    }
    result
}
/// Independent output rows share input loads; accumulation order stays unchanged.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_four_rows<const R: usize>(
    packed: &[f32],
    weights: &[f32],
    cols: usize,
) -> [[f32; 4]; R] {
    use core::arch::wasm32::*;
    let mut acc = [f32x4_splat(0.); R];
    let row_ptrs: [*const f32; R] = core::array::from_fn(|r| weights.as_ptr().add(r * cols));
    let mut input_ptr = packed.as_ptr();
    for c in 0..cols {
        let input = v128_load(input_ptr.cast());
        input_ptr = input_ptr.add(4);
        macro_rules! row {($($r:literal),*)=>{$(if R>$r {
            acc[$r]=f32x4_add(acc[$r],f32x4_mul(input,f32x4_splat(*row_ptrs[$r].add(c))));
        })*};}
        row!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
    let mut out = [[0.; 4]; R];
    for r in 0..R {
        v128_store(out[r].as_mut_ptr().cast(), acc[r]);
    }
    out
}
/// Share each weight broadcast across independent token vectors. Each lane
/// still visits columns in scalar order, with separate F32 multiply and add.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn matrix_multi<const G: usize>(
    x: &[f32],
    w: &[f32],
    n: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let tokens = G * 4;
    let mut out = vec![0.0; n * rows];
    let mut packed = vec![0.0; cols * tokens];
    let mut t = 0;
    while t + tokens <= n {
        for c in 0..cols {
            for lane in 0..tokens {
                packed[c * tokens + lane] = x[(t + lane) * cols + c];
            }
        }
        let mut r = 0;
        while r + 16 <= rows {
            let sums = dot_multi::<16, G>(&packed, &w[r * cols..(r + 16) * cols], cols);
            for j in 0..16 {
                for lane in 0..tokens {
                    out[(t + lane) * rows + r + j] = sums[j][lane / 4][lane % 4];
                }
            }
            r += 16;
        }
        while r < rows {
            let sums = dot_multi::<1, G>(&packed, &w[r * cols..(r + 1) * cols], cols);
            for lane in 0..tokens {
                out[(t + lane) * rows + r] = sums[0][lane / 4][lane % 4];
            }
            r += 1;
        }
        t += tokens;
    }
    // Keep the four-lane fast path for the final complete vector.
    while t + 4 <= n {
        for c in 0..cols {
            for lane in 0..4 {
                packed[c * 4 + lane] = x[(t + lane) * cols + c];
            }
        }
        let mut r = 0;
        while r + 16 <= rows {
            let sums = dot_four_rows::<16>(&packed, &w[r * cols..(r + 16) * cols], cols);
            for j in 0..16 {
                for lane in 0..4 {
                    out[(t + lane) * rows + r + j] = sums[j][lane];
                }
            }
            r += 16;
        }
        while r < rows {
            let sums = dot_four(&packed, &w[r * cols..(r + 1) * cols]);
            for lane in 0..4 {
                out[(t + lane) * rows + r] = sums[lane];
            }
            r += 1;
        }
        t += 4;
    }
    while t < n {
        for r in 0..rows {
            let mut sum = 0.0;
            for c in 0..cols {
                sum += x[t * cols + c] * w[r * cols + c];
            }
            out[t * rows + r] = sum;
        }
        t += 1;
    }
    out
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_multi<const R: usize, const G: usize>(
    packed: &[f32],
    weights: &[f32],
    cols: usize,
) -> [[[f32; 4]; G]; R] {
    use core::arch::wasm32::*;
    let mut acc = [[f32x4_splat(0.); G]; R];
    let row_ptrs: [*const f32; R] = core::array::from_fn(|r| weights.as_ptr().add(r * cols));
    let mut input_ptr = packed.as_ptr();
    for c in 0..cols {
        let inputs: [v128; G] = core::array::from_fn(|g| v128_load(input_ptr.add(g * 4).cast()));
        input_ptr = input_ptr.add(G * 4);
        macro_rules! row {($($r:literal),*)=>{$(if R>$r {
            let weight = f32x4_splat(*row_ptrs[$r].add(c));
            for g in 0..G { acc[$r][g] = f32x4_add(acc[$r][g],f32x4_mul(inputs[g],weight)); }
        })*};}
        row!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
    let mut out = [[[0.; 4]; G]; R];
    for r in 0..R {
        for g in 0..G {
            v128_store(out[r][g].as_mut_ptr().cast(), acc[r][g]);
        }
    }
    out
}
pub fn lora_project(
    x: &[f32],
    base: &[f32],
    a: &[f32],
    b: &[f32],
    n: usize,
    rows: usize,
    cols: usize,
    rank: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    lora_project_limit(x, base, a, b, n, rows, cols, rank, scale, FUSED_WORK_LIMIT)
}
fn lora_project_limit(
    x: &[f32],
    base: &[f32],
    a: &[f32],
    b: &[f32],
    n: usize,
    rows: usize,
    cols: usize,
    rank: usize,
    scale: f32,
    limit: u64,
) -> Result<Vec<f32>> {
    if n == 0
        || rows == 0
        || cols == 0
        || rank == 0
        || rank > 256
        || [n, rows, cols].iter().any(|&d| d > 262144)
        || n.checked_mul(rows).is_none_or(|v| v > MAX_FLOATS)
        || n.checked_mul(rank).is_none_or(|v| v > MAX_FLOATS)
    {
        return Err("projection dimensions/output bounds".into());
    }
    let work = (n as u64) * (rows as u64) * (cols as u64)
        + (n as u64) * (rank as u64) * ((cols + rows) as u64);
    if work > limit || !scale.is_finite() {
        return Err("projection work/scale limit".into());
    }
    let y = matrix(x, base, n, rows, cols)?;
    let z = matrix(&matrix(x, a, n, rank, cols)?, b, n, rows, rank)?;
    let result: Vec<f32> = y
        .into_iter()
        .zip(z)
        .map(|(y, z)| bf(bf(y) + bf(scale * z)))
        .collect();
    if !result.iter().all(|v| v.is_finite()) {
        return Err("invalid projection output".into());
    }
    Ok(result)
}
fn bf(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}
// MLX Metal's BF16 Sigmoid uses symmetric exp(abs(x)) and BF16 intermediates.
fn bf_sigmoid(x: f32) -> f32 {
    let y = bf(1.0 / bf(1.0 + bf(x.abs().exp())));
    if x < 0. {
        y
    } else {
        bf(1.0 - y)
    }
}
fn bf_silu(x: f32) -> f32 {
    bf(x * bf_sigmoid(x))
}
fn bf_softplus(x: f32) -> f32 {
    bf(x.max(0.) + bf(bf((-x.abs()).exp()).ln_1p()))
}
pub fn delta(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    dk: usize,
    dv: usize,
) -> Result<Vec<f32>> {
    let n = g.len();
    if n == 0
        || dk == 0
        || dv == 0
        || q.len() != n * dk
        || k.len() != q.len()
        || v.len() != n * dv
        || beta.len() != n
        || state.len() != dk * dv
        || !g.iter().all(|x| (0.0..=1.0).contains(x))
        || !beta.iter().all(|x| (0.0..=1.0).contains(x))
    {
        return Err("delta shape/gate".into());
    }
    #[cfg(target_arch = "wasm32")]
    if dv % 4 == 0 {
        return Ok(unsafe { delta_simd::run(q, k, v, g, beta, state, dk, dv) });
    }
    let mut y = vec![0.0; n * dv];
    for t in 0..n {
        for d in 0..dv {
            let s = &mut state[d * dk..(d + 1) * dk];
            let mut mem = 0.0;
            for i in 0..dk {
                s[i] *= g[t];
                mem += s[i] * k[t * dk + i];
            }
            let update = (v[t * dv + d] - mem) * beta[t];
            let mut o = 0.0;
            for i in 0..dk {
                s[i] += k[t * dk + i] * update;
                o += s[i] * q[t * dk + i];
            }
            y[t * dv + d] = o;
        }
    }
    Ok(y)
}
/// Evaluate one bounded primitive. `weight` is only the requested stable-memory tile.
pub fn execute(r: &Request, x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    let d = &r.dims;
    // Flat binary element operations fit 900k input floats (two operands),
    // even when their single length exceeds the multi-axis dimension bound.
    let flat_pair = matches!(r.op.as_str(), "add_bf16" | "swiglu_bf16" | "attention_gate")
        && d.len() == 1
        && d[0] > 0
        && d[0] <= MAX_FLOATS / 2
        && d[0].checked_mul(2) == Some(x.len());
    if !flat_pair && d.iter().any(|&v| v > 262144) {
        return Err("dimension limit".into());
    }
    let y = match r.op.as_str() {
        "matmul" | "matmul_reference" | "linear_bf16" if d.len() == 3 || d.len() == 4 => {
            if (d[0] as u64) * (d[1] as u64) * (d[2] as u64) > 120_000_000 {
                return Err("matmul work limit".into());
            }
            if r.op == "matmul_reference" {
                matrix_reference(x, weight, d[0], d[1], d[2])?
            } else {
                let y = matrix(x, weight, d[0], d[1], d[2])?;
                if r.op == "linear_bf16" {
                    y.into_iter().map(bf).collect()
                } else {
                    y
                }
            }
        }
        "add_norm_bf16"
            if d.len() == 2
                && d[0] > 0
                && d[1] > 0
                && d[0].checked_mul(d[1]).is_some_and(|count| {
                    count <= MAX_FLOATS / 2 && count.checked_mul(2) == Some(x.len())
                })
                && weight.len() == d[1]
                && r.scalars.len() == 1
                && r.scalars[0].is_finite()
                && r.scalars[0] > 0. =>
        {
            let count = d[0] * d[1];
            let mut values: Vec<_> = (0..count).map(|i| bf(x[i] + x[count + i])).collect();
            let mut normalized = Vec::with_capacity(count);
            for row in values.chunks_exact(d[1]) {
                normalized.extend(rms(row, weight, r.scalars[0]).into_iter().map(bf));
            }
            values.extend(normalized);
            values
        }
        "rms_bf16" | "rms_scaled" | "gated_norm"
            if d.len() == 2
                && d[1] > 0
                && d[0] * d[1]
                    == (if r.op == "gated_norm" {
                        x.len() / 2
                    } else {
                        x.len()
                    })
                && (weight.len() == d[1] || (r.op == "rms_scaled" && weight.is_empty()))
                && !r.scalars.is_empty()
                && r.scalars[0] > 0. =>
        {
            if (r.op == "rms_scaled" && r.scalars.len() != 2)
                || (r.op == "gated_norm" && x.len() != 2 * d[0] * d[1])
            {
                return Err("normalization shape".into());
            }
            let ones;
            let w = if weight.is_empty() {
                ones = vec![1.; d[1]];
                &ones
            } else {
                weight
            };
            let mut result = Vec::with_capacity(d[0] * d[1]);
            for (i, row) in x[..d[0] * d[1]].chunks_exact(d[1]).enumerate() {
                let values = rms(row, w, r.scalars[0]);
                for (j, v) in values.into_iter().enumerate() {
                    let v = bf(v);
                    result.push(if r.op == "rms_scaled" {
                        bf(v * bf(r.scalars[1]))
                    } else if r.op == "gated_norm" {
                        bf(v * silu(x[d[0] * d[1] + i * d[1] + j]))
                    } else {
                        v
                    });
                }
            }
            result
        }
        "add_bf16" | "swiglu_bf16" | "attention_gate" if d.len() == 1 && x.len() == 2 * d[0] => (0
            ..d[0])
            .map(|i| match r.op.as_str() {
                "add_bf16" => bf(x[i] + x[d[0] + i]),
                "swiglu_bf16" => bf(bf_silu(x[i]) * x[d[0] + i]),
                _ => bf(x[i] * bf_sigmoid(x[d[0] + i])),
            })
            .collect(),
        "delta_gates"
            if d.len() == 2
                && d[0] > 0
                && d[1] > 0
                && x.len() == 2 * d[0] * d[1]
                && weight.len() == 2 * d[1] =>
        {
            let count = d[0] * d[1];
            let mut y = vec![0.; 2 * count];
            for i in 0..count {
                y[i] = (-weight[i % d[1]].exp() * bf_softplus(bf(x[i] + weight[d[1] + i % d[1]])))
                    .exp();
                y[count + i] = bf_sigmoid(x[count + i]);
            }
            y
        }
        "rope"
            if d.len() == 4
                && d[0] > 0
                && d[1] > 0
                && d[2] > 0
                && d[2] <= d[1]
                && d[2] % 2 == 0
                && x.len() == d[0] * d[1]
                && r.scalars.len() == 1
                && r.scalars[0] > 0. =>
        {
            let mut y = x.to_vec();
            let half = d[2] / 2;
            for t in 0..d[0] {
                for i in 0..half {
                    let angle =
                        (t + d[3]) as f32 * r.scalars[0].powf(-((2 * i) as f32) / d[2] as f32);
                    let (sin, cos) = angle.sin_cos();
                    let a = x[t * d[1] + i];
                    let b = x[t * d[1] + half + i];
                    y[t * d[1] + i] = bf(a * cos - b * sin);
                    y[t * d[1] + half + i] = bf(b * cos + a * sin);
                }
            }
            y
        }
        "conv_state"
            if d.len() == 4
                && d[0] > 0
                && d[1] > 0
                && (1..=8).contains(&d[2])
                && x.len() == (d[0] + d[2] - 1) * d[1]
                && weight.len() == d[1] * d[2] =>
        {
            let (n, c, k) = (d[0], d[1], d[2]);
            let mut y = vec![0.; n * c];
            for t in 0..n {
                for ch in 0..c {
                    let mut sum = 0.;
                    for j in 0..k {
                        sum += x[(t + j) * c + ch] * weight[ch * k + j];
                    }
                    y[t * c + ch] = bf_silu(bf(sum));
                }
            }
            y.extend_from_slice(&x[x.len() - (k - 1) * c..]);
            y
        }
        "rms"
            if d.len() == 2
                && d[1] > 0
                && d[0] * d[1] == x.len()
                && weight.len() == d[1]
                && r.scalars.len() == 1
                && r.scalars[0] > 0.0 =>
        {
            x.chunks_exact(d[1])
                .flat_map(|row| rms(row, weight, r.scalars[0]))
                .collect()
        }
        "bf16" if d.is_empty() => x
            .iter()
            .map(|v| {
                let b = v.to_bits();
                f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
            })
            .collect(),
        "lora_finish" if d.len() == 1 && x.len() == 2 * d[0] && r.scalars.len() == 1 => {
            fn bf(v: f32) -> f32 {
                let b = v.to_bits();
                f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
            }
            (0..d[0])
                .map(|i| bf(bf(x[i]) + bf(r.scalars[0] * x[d[0] + i])))
                .collect()
        }
        "silu" if d.is_empty() => x.iter().map(|&v| silu(v)).collect(),
        "softmax" if d.is_empty() && r.scalars.len() == 1 => softmax(x, r.scalars[0])?,
        "add" | "multiply" if d.len() == 1 && x.len() == 2 * d[0] => (0..d[0])
            .map(|i| {
                if r.op == "add" {
                    x[i] + x[i + d[0]]
                } else {
                    x[i] * x[i + d[0]]
                }
            })
            .collect(),
        "conv"
            if d.len() == 3
                && d[2] > 0
                && d[2] <= 8
                && x.len() == d[0] * d[1]
                && weight.len() == d[1] * d[2] =>
        {
            let (n, c, k) = (d[0], d[1], d[2]);
            let mut out = vec![0.0; x.len()];
            for t in 0..n {
                for ch in 0..c {
                    let mut v = 0.0;
                    for j in 0..k {
                        if t + j + 1 >= k {
                            v += x[(t + j + 1 - k) * c + ch] * weight[ch * k + j];
                        }
                    }
                    out[t * c + ch] = silu(v);
                }
            }
            out
        }
        "delta_heads_bf16" if d.len() == 4 => {
            let (n, dk, dv, h) = (d[0], d[1], d[2], d[3]);
            int8_prefix(r, x.len())?;
            let activ = n * (2 * dk + dv);
            let tail = 2 * n + dk * dv;
            if x.len() != h * (activ + tail) {
                return Err("delta heads input".into());
            }
            let mut out = Vec::with_capacity(h * (n * dv + dk * dv));
            let mut states = Vec::with_capacity(h * dk * dv);
            for head in 0..h {
                let a = &x[head * activ..(head + 1) * activ];
                let z = &x[h * activ + head * tail..h * activ + (head + 1) * tail];
                let mut state = z[2 * n..].to_vec();
                let y = delta(
                    &a[..n * dk],
                    &a[n * dk..2 * n * dk],
                    &a[2 * n * dk..],
                    &z[..n],
                    &z[n..2 * n],
                    &mut state,
                    dk,
                    dv,
                )?;
                out.extend(y.into_iter().map(bf));
                states.extend(state);
            }
            out.extend(states);
            out
        }
        "delta" | "delta_bf16" if d.len() == 3 => {
            let (n, dk, dv) = (d[0], d[1], d[2]);
            if n == 0
                || dk == 0
                || dv == 0
                || n > 512
                || dk > 128
                || dv > 128
                || x.len() != 2 * n * dk + n * dv + 2 * n + dk * dv
            {
                return Err("delta limits".into());
            }
            let (q, z) = x.split_at(n * dk);
            let (k, z) = z.split_at(n * dk);
            let (v, z) = z.split_at(n * dv);
            let (g, z) = z.split_at(n);
            let (b, s) = z.split_at(n);
            let mut state = s.to_vec();
            let mut y = delta(q, k, v, g, b, &mut state, dk, dv)?;
            if r.op == "delta_bf16" {
                for v in &mut y {
                    *v = bf(*v);
                }
            }
            y.extend(state);
            y
        }
        "attention_heads_bf16" if d.len() == 3 => {
            let (n, width, heads) = (d[0], d[1], d[2]);
            if n == 0
                || n > 512
                || width == 0
                || width > 256
                || heads == 0
                || heads > 8
                || x.len() != 3 * n * width * heads
            {
                return Err("attention heads bounds".into());
            }
            let mut single = r.clone();
            single.op = "attention_bf16".into();
            single.dims = vec![n, width];
            let mut out = Vec::with_capacity(n * width * heads);
            for head in 0..heads {
                out.extend(execute(
                    &single,
                    &x[head * 3 * n * width..(head + 1) * 3 * n * width],
                    &[],
                )?);
            }
            out
        }
        "rope_heads" if d.len() == 5 => {
            let heads = d[4];
            if d[0] == 0
                || d[0] > 512
                || d[1] == 0
                || d[1] > 256
                || heads == 0
                || heads > 16
                || x.len() != d[0] * d[1] * heads
            {
                return Err("rope heads bounds".into());
            }
            let mut single = r.clone();
            single.op = "rope".into();
            single.dims = d[..4].to_vec();
            let mut out = Vec::with_capacity(x.len());
            for head in 0..heads {
                out.extend(execute(
                    &single,
                    &x[head * d[0] * d[1]..(head + 1) * d[0] * d[1]],
                    &[],
                )?);
            }
            out
        }
        "attention_reference" | "attention_bf16_reference"
            if d.len() == 2
                && d[0] > 0
                && d[0] <= 512
                && d[1] > 0
                && d[1] <= 256
                && x.len() == 3 * d[0] * d[1] =>
        {
            let (n, h) = (d[0], d[1]);
            let (q, z) = x.split_at(n * h);
            let (k, v) = z.split_at(n * h);
            let mut out = vec![0.0; n * h];
            for t in 0..n {
                let mut scores = vec![0.0; t + 1];
                for s in 0..=t {
                    scores[s] = (0..h).map(|i| q[t * h + i] * k[s * h + i]).sum::<f32>()
                        / (h as f32).sqrt();
                    if r.op == "attention_bf16_reference" {
                        scores[s] = bf(scores[s]);
                    }
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - m).exp();
                }
                let z = scores.iter().sum::<f32>();
                for i in 0..h {
                    out[t * h + i] = (0..=t)
                        .map(|s| {
                            let p = scores[s] / z;
                            let p = if r.op == "attention_bf16_reference" {
                                bf(p)
                            } else {
                                p
                            };
                            p * v[s * h + i]
                        })
                        .sum();
                }
            }
            if r.op == "attention_bf16_reference" {
                for v in &mut out {
                    *v = bf(*v);
                }
            }
            out
        }
        "attention" | "attention_bf16"
            if d.len() == 2
                && d[0] > 0
                && d[0] <= 512
                && d[1] > 0
                && d[1] <= 256
                && x.len() == 3 * d[0] * d[1] =>
        {
            let (n, h) = (d[0], d[1]);
            let (q, z) = x.split_at(n * h);
            let (k, v) = z.split_at(n * h);
            let mut out = vec![0.0; n * h];
            for t in 0..n {
                let mut scores = vec![0.0; t + 1];
                for s in 0..=t {
                    scores[s] = (0..h).map(|i| q[t * h + i] * k[s * h + i]).sum::<f32>()
                        / (h as f32).sqrt();
                    if r.op == "attention_bf16" {
                        scores[s] = bf(scores[s]);
                    }
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - m).exp();
                }
                let z = scores.iter().sum::<f32>();
                let probabilities = scores
                    .iter()
                    .map(|&v| {
                        let p = v / z;
                        if r.op == "attention_bf16" {
                            bf(p)
                        } else {
                            p
                        }
                    })
                    .collect::<Vec<_>>();
                #[cfg(target_arch = "wasm32")]
                if h % 4 == 0 {
                    // Shape guard above proves every four-lane value load and output store.
                    unsafe {
                        attention_value_lanes(&probabilities, v, h, &mut out[t * h..(t + 1) * h]);
                    }
                    continue;
                }
                for i in 0..h {
                    out[t * h + i] = probabilities
                        .iter()
                        .enumerate()
                        .map(|(s, &p)| p * v[s * h + i])
                        .sum();
                }
            }
            if r.op == "attention_bf16" {
                for v in &mut out {
                    *v = bf(*v);
                }
            }
            out
        }
        _ => return Err("unsupported operation or shape".into()),
    };
    if y.len() > MAX_FLOATS || !y.iter().all(|v| v.is_finite()) {
        return Err("invalid output".into());
    }
    Ok(y)
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn attention_value_lanes(p: &[f32], v: &[f32], h: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    for i in (0..h).step_by(4) {
        let mut sum = f32x4_splat(-0.);
        for (s, &weight) in p.iter().enumerate() {
            sum = f32x4_add(
                sum,
                f32x4_mul(
                    f32x4_splat(weight),
                    v128_load(v.as_ptr().add(s * h + i).cast()),
                ),
            );
        }
        v128_store(out.as_mut_ptr().add(i).cast(), sum);
    }
}
pub fn unpack(t: &Tensor, b: &[u8]) -> Result<Vec<f32>> {
    let count = t.rows.checked_mul(t.cols).ok_or("overflow")?;
    if count > 30_000_000 {
        return Err("tile too large".into());
    }
    match t.dtype.as_str() {
        "f32" if b.len() == count * 4 => Ok(b
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()),
        "bf16" if b.len() == count * 2 => Ok(b
            .chunks_exact(2)
            .map(|v| f32::from_bits((u16::from_le_bytes(v.try_into().unwrap()) as u32) << 16))
            .collect()),
        "int8" if b.len() == count + t.rows * 4 => {
            let mut y = vec![0.0; count];
            for r in 0..t.rows {
                let scale =
                    f32::from_le_bytes(b[count + r * 4..count + r * 4 + 4].try_into().unwrap());
                if !scale.is_finite() || scale <= 0.0 {
                    return Err("invalid scale".into());
                }
                for c in 0..t.cols {
                    y[r * t.cols + c] = (b[r * t.cols + c] as i8) as f32 * scale;
                }
            }
            Ok(y)
        }
        _ => Err("tensor encoding".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delta_update_and_continuation() {
        let mut s = vec![0.0; 4];
        let y = delta(
            &[1., 0., 0., 1.],
            &[1., 0., 0., 1.],
            &[2., 3., 4., 5.],
            &[1., 0.5],
            &[1., 1.],
            &mut s,
            2,
            2,
        )
        .unwrap();
        assert_eq!(y, vec![2., 3., 4., 5.]);
        assert_eq!(s, vec![1., 4., 1.5, 5.]);
    }
    #[test]
    fn valid_unknown_distribution() {
        let p = softmax(&[0., 1., 2.], 1.30515697).unwrap();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1]);
        assert!(softmax(&[f32::NAN, 1.], 1.).is_err());
    }
    #[test]
    fn checksum_and_identity() {
        let r = Request {
            version: 1,
            model: "a".repeat(64),
            pack_hash: "c".repeat(64),
            input_hash: "b".repeat(64),
            step: 0,
            op: "silu".into(),
            tensor: String::new(),
            dims: vec![],
            scalars: vec![],
            aux: vec![],
            encoding: String::new(),
        };
        let mut b = encode(&r, &[1., 2.]).unwrap();
        assert_eq!(decode(&b).unwrap().1, vec![1., 2.]);
        b[5] ^= 1;
        assert!(decode(&b).is_err());
    }
    #[test]
    fn matrix_reference() {
        assert_eq!(
            matrix(&[1., 2.], &[3., 4., 5., 6.], 1, 2, 2).unwrap(),
            vec![11., 17.]
        );
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub value: Option<String>,
    pub probabilities: Vec<f32>,
    pub unknown_probability: f32,
    pub abstained: bool,
    pub raw_logits: Vec<f32>,
}
pub fn decide(options: &[String], all_logits: &[f32], temperature: f32) -> Result<Decision> {
    if all_logits.len() != 256 || options.len() > 7 {
        return Err("readout bounds".into());
    }
    decide_candidates(options, &all_logits[..options.len() + 1], temperature)
}
pub fn decide_candidates(
    options: &[String],
    all_logits: &[f32],
    temperature: f32,
) -> Result<Decision> {
    if !(2..=7).contains(&options.len())
        || all_logits.len() != options.len() + 1
        || options
            .iter()
            .any(|s| s.is_empty() || s.len() > 128 || s == "__unknown__")
        || options
            .iter()
            .enumerate()
            .any(|(i, s)| options[..i].contains(s))
    {
        return Err("choice bounds/uniqueness".into());
    }
    let logits = &all_logits[..options.len() + 1];
    let p = softmax(logits, temperature)?;
    let mut top = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[top] {
            top = i;
        }
    }
    let abstained = top == options.len();
    Ok(Decision {
        value: if abstained {
            None
        } else {
            Some(options[top].clone())
        },
        probabilities: p[..options.len()].to_vec(),
        unknown_probability: p[options.len()],
        abstained,
        raw_logits: logits.to_vec(),
    })
}
/// Read only a contiguous output-row tile. INT8 row scales are read separately.
pub fn load_weight<F>(t: &Tensor, r: &Request, mut read: F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<Vec<u8>>,
{
    let (start, rows) = if matches!(
        r.op.as_str(),
        "matmul" | "matmul_reference" | "linear_bf16" | "lora_project" | "conv_state"
    ) && r.dims.len() == 4
    {
        if r.dims[2] != t.cols
            || r.dims[1] == 0
            || r.dims[3].checked_add(r.dims[1]).is_none()
            || r.dims[3] + r.dims[1] > t.rows
        {
            return Err("tile shape/range".into());
        }
        (r.dims[3], r.dims[1])
    } else {
        (0, t.rows)
    };
    let count = rows.checked_mul(t.cols).ok_or("tile overflow")?;
    let bytes_per = match t.dtype.as_str() {
        "f32" => 4,
        "bf16" => 2,
        "int8" => 1,
        _ => return Err("tensor precision".into()),
    };
    let mut n = count.checked_mul(bytes_per).ok_or("tile bytes")?;
    if n > 128 * 1024 * 1024 || count > 30_000_000 {
        return Err("tile limit".into());
    }
    let expected = t
        .rows
        .checked_mul(t.cols)
        .and_then(|v| v.checked_mul(bytes_per))
        .and_then(|v| v.checked_add(if t.dtype == "int8" { t.rows * 4 } else { 0 }))
        .ok_or("tensor overflow")?;
    if expected as u64 != t.bytes {
        return Err("manifest tensor size".into());
    }
    let mut b = read(t.offset + (start * t.cols * bytes_per) as u64, n)?;
    if t.dtype == "int8" {
        let scales = read(t.offset + (t.rows * t.cols + start * 4) as u64, rows * 4)?;
        n += scales.len();
        b.extend(scales);
    }
    let tile = Tensor {
        name: t.name.clone(),
        offset: 0,
        rows,
        cols: t.cols,
        dtype: t.dtype.clone(),
        bytes: n as u64,
    };
    let w = unpack(&tile, &b)?;
    if !w.iter().all(|v| v.is_finite()) {
        return Err("nonfinite weight".into());
    }
    Ok((w, n as u64))
}
#[cfg(test)]
mod contract_tests {
    use super::*;
    fn request() -> Request {
        Request {
            version: 1,
            model: "a".repeat(64),
            pack_hash: "b".repeat(64),
            input_hash: "c".repeat(64),
            step: 7,
            op: "matmul".into(),
            tensor: "weights".into(),
            dims: vec![1, 1, 2, 1],
            scalars: vec![],
            aux: vec![],
            encoding: String::new(),
        }
    }
    #[test]
    fn nonzero_int8_tile_reads_corresponding_scale() {
        let t = Tensor {
            name: "weights".into(),
            offset: 0,
            rows: 2,
            cols: 2,
            dtype: "int8".into(),
            bytes: 12,
        };
        let mut bytes = vec![1, 2, 3, 4];
        bytes.extend(0.5f32.to_le_bytes());
        bytes.extend(2.0f32.to_le_bytes());
        let (w, n) = load_weight(&t, &request(), |offset, n| {
            Ok(bytes[offset as usize..offset as usize + n].to_vec())
        })
        .unwrap();
        assert_eq!(w, vec![6., 8.]);
        assert_eq!(n, 6);
        assert_eq!(execute(&request(), &[1., 2.], &w).unwrap(), vec![22.]);
    }
    #[test]
    fn decision_unknown_is_at_candidate_count() {
        let opts = vec!["yes".into(), "no".into()];
        let mut logits = vec![-100.; 256];
        logits[2] = 5.;
        let d = decide(&opts, &logits, 1.30515697).unwrap();
        assert!(d.abstained);
        assert!(d.value.is_none());
        assert!(d.unknown_probability > 0.999);
        logits[0] = 6.;
        assert_eq!(
            decide(&opts, &logits, 1.).unwrap().value,
            Some("yes".into())
        );
        assert!(decide(&["yes".into(), "yes".into()], &logits, 1.).is_err());
    }
    #[test]
    fn invalid_shapes_return_errors_without_panicking() {
        let mut r = request();
        r.op = "rms".into();
        r.dims = vec![1, 0];
        r.scalars = vec![1e-6];
        assert!(execute(&r, &[], &[]).is_err());
        r.op = "matmul".into();
        r.dims = vec![usize::MAX, 2, 2];
        assert!(execute(&r, &[], &[]).is_err());
    }
}
#[cfg(test)]
mod optimization_tests {
    use super::*;
    #[test]
    fn token_tiles_preserve_every_scalar_dot_including_tail() {
        for n in [1, 3, 4, 5, 8, 9] {
            let x: Vec<f32> = (0..n * 13)
                .map(|i| ((i * 37 % 101) as f32 - 50.) / 17.)
                .collect();
            let w: Vec<f32> = (0..7 * 13)
                .map(|i| ((i * 71 % 97) as f32 - 48.) / 23.)
                .collect();
            let scalar = matrix_reference(&x, &w, n, 7, 13).unwrap();
            let tiled = matrix(&x, &w, n, 7, 13).unwrap();
            assert_eq!(
                scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                tiled.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn fused_projection_preserves_bf16_rounding_boundaries() {
        assert!(lora_project(&[], &[], &[], &[], usize::MAX, 1, 1, 1, 2.).is_err());
        assert!(lora_project(&[f32::MAX], &[2.], &[1.], &[1.], 1, 1, 1, 1, 2.).is_err());
        let (n, rows, cols, rank) = (5, 7, 13, 3);
        let x: Vec<_> = (0..n * cols).map(|i| (i as f32 - 20.) / 31.).collect();
        let base: Vec<_> = (0..rows * cols).map(|i| (i as f32 - 30.) / 17.).collect();
        let a: Vec<_> = (0..rank * cols).map(|i| (i as f32 - 10.) / 41.).collect();
        let b: Vec<_> = (0..rows * rank).map(|i| (i as f32 - 5.) / 13.).collect();
        let y = matrix_reference(&x, &base, n, rows, cols).unwrap();
        let z = matrix_reference(
            &matrix_reference(&x, &a, n, rank, cols).unwrap(),
            &b,
            n,
            rows,
            rank,
        )
        .unwrap();
        let expected: Vec<_> = y
            .iter()
            .zip(z)
            .map(|(y, z)| bf(bf(*y) + bf(2. * z)))
            .collect();
        assert_eq!(
            lora_project(&x, &base, &a, &b, n, rows, cols, rank, 2.).unwrap(),
            expected
        );
        for count in 2..=7 {
            let options: Vec<_> = (0..count).map(|i| i.to_string()).collect();
            let logits: Vec<_> = (0..256).map(|i| i as f32).collect();
            assert_eq!(
                decide(&options, &logits, 1.30515697).unwrap().raw_logits,
                decide_candidates(&options, &logits[..count + 1], 1.30515697)
                    .unwrap()
                    .raw_logits
            );
        }
    }
}

fn evaluate_integer<F>(
    r: &Request,
    x: &[f32],
    m: &Manifest,
    mut read: &mut F,
    prepared: Option<&int8_kernel::QuantizedRows>,
) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<Vec<u8>>,
{
    if r.dims.len() != 4 {
        return Err("integer matmul dims".into());
    }
    let (n, rows, cols, start) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3]);
    if n == 0
        || n > 512
        || rows == 0
        || rows % 8 != 0
        || cols == 0
        || cols % 256 != 0
        || rows.checked_mul(cols).is_none_or(|v| v > 30_000_000)
        || n.checked_mul(rows).is_none_or(|v| v > MAX_FLOATS)
        || n.checked_mul(cols) != Some(x.len())
        || (n as u64) * (rows as u64) * (cols as u64) > 4_000_000_000
    {
        return Err("integer matmul bounds".into());
    }
    let tensor = m
        .tensors
        .iter()
        .find(|t| t.name == r.tensor)
        .ok_or("integer tensor")?;
    if tensor.dtype != "int8"
        || tensor.cols != cols
        || start.checked_add(rows).is_none_or(|v| v > tensor.rows)
    {
        return Err("integer tensor shape".into());
    }
    let with_lora = r.op == "lora_integer";
    let lora = if with_lora {
        if r.aux.len() != 2 || r.scalars.len() != 1 || !r.scalars[0].is_finite() {
            return Err("integer LoRA metadata".into());
        }
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[0])
            .ok_or("integer LoRA A")?;
        let b = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[1])
            .ok_or("integer LoRA B")?;
        if a.rows == 0
            || a.rows > 256
            || a.cols != cols
            || b.cols != a.rows
            || start.checked_add(rows).is_none_or(|v| v > b.rows)
        {
            return Err("integer LoRA shape".into());
        }
        let work = n as u64 * (rows as u64 * cols as u64 + a.rows as u64 * (rows + cols) as u64);
        if work > 4_000_000_000 {
            return Err("integer LoRA work limit".into());
        }
        Some((a, b))
    } else {
        if !r.aux.is_empty() || !r.scalars.is_empty() {
            return Err("integer matmul metadata".into());
        }
        None
    };
    let values = profile::measure("base_stable_read", || {
        read(tensor.offset + (start * cols) as u64, rows * cols)
    })?;
    let scale_bytes = read(
        tensor.offset + (tensor.rows * cols + start * 4) as u64,
        rows * 4,
    )?;
    let scales = scale_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let weights = values.into_iter().map(|v| v as i8).collect::<Vec<_>>();
    let owned;
    let q = if let Some(q) = prepared {
        q
    } else {
        owned = profile::measure("activation_quantize", || {
            int8_kernel::quantize_rows(x, n, cols)
        })?;
        &owned
    };
    let base = profile::measure("base_project_inclusive", || {
        int8_kernel::project(&q, &weights, &scales, rows)
    })?;
    let mut read_bytes = (rows * cols + rows * 4) as u64;
    let out = if let Some((a, b)) = lora {
        let mut ar = r.clone();
        ar.op = "matmul".into();
        ar.dims = vec![n, a.rows, cols];
        let (aw, ra) = profile::measure("lora_load_A", || load_weight(a, &ar, &mut read))?;
        ar.dims = vec![n, rows, a.rows, start];
        let (bw, rb) = profile::measure("lora_load_B", || load_weight(b, &ar, &mut read))?;
        read_bytes += ra + rb;
        let ax = profile::measure("lora_matmul_A", || matrix(x, &aw, n, a.rows, cols))?;
        let z = profile::measure("lora_matmul_B", || matrix(&ax, &bw, n, rows, a.rows))?;
        base.into_iter()
            .zip(z)
            .map(|(v, z)| bf(bf(v) + bf(r.scalars[0] * z)))
            .collect::<Vec<_>>()
    } else if r.op == "linear_integer_bf16" {
        base.into_iter().map(bf).collect()
    } else {
        base
    };
    if !out.iter().all(|v| v.is_finite()) {
        return Err("integer fused output".into());
    }
    return Ok((out, read_bytes));
}
fn evaluate_mlp<F>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<Vec<u8>>,
{
    if r.dims.len() != 4
        || r.aux.len() != 1
        || r.scalars.len() != 1
        || !r.scalars[0].is_finite()
        || !r.tensor.ends_with(".mlp.gate_proj.weight")
        || r.aux[0] != r.tensor.replace(".gate_proj.weight", ".up_proj.weight")
    {
        return Err("fused MLP metadata".into());
    }
    let (n, rows, cols, start) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3]);
    if n == 0
        || n > 512
        || rows == 0
        || rows % 8 != 0
        || cols == 0
        || cols % 256 != 0
        || n.checked_mul(cols) != Some(x.len())
        || n.checked_mul(rows).is_none_or(|v| v > MAX_FLOATS)
        || rows.checked_mul(cols).is_none_or(|v| v > 30_000_000)
    {
        return Err("fused MLP bounds".into());
    }
    let mut requests = Vec::new();
    let mut work = 0u64;
    for name in [&r.tensor, &r.aux[0]] {
        let prefix = name
            .strip_suffix(".weight")
            .ok_or("fused MLP weight name")?;
        let mut request = r.clone();
        request.op = "lora_integer".into();
        request.tensor = name.clone();
        request.aux = vec![
            format!("{prefix}.lora_A.weight"),
            format!("{prefix}.lora_B.weight"),
        ];
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == request.aux[0])
            .ok_or("fused MLP LoRA A")?;
        let b = m
            .tensors
            .iter()
            .find(|t| t.name == request.aux[1])
            .ok_or("fused MLP LoRA B")?;
        let base = m
            .tensors
            .iter()
            .find(|t| t.name == *name)
            .ok_or("fused MLP base")?;
        if a.rows == 0
            || a.rows > 256
            || a.cols != cols
            || b.cols != a.rows
            || base.dtype != "int8"
            || base.cols != cols
            || start
                .checked_add(rows)
                .is_none_or(|v| v > base.rows || v > b.rows)
        {
            return Err("fused MLP weight shape".into());
        }
        work += n as u64 * (rows as u64 * cols as u64 + a.rows as u64 * (rows + cols) as u64);
        requests.push(request);
    }
    if work > 4_000_000_000 {
        return Err("fused MLP work limit".into());
    }
    let q = profile::measure("activation_quantize", || {
        int8_kernel::quantize_rows(x, n, cols)
    })?;
    let (gate, rg) = evaluate_integer(&requests[0], x, m, read, Some(&q))?;
    let (up, ru) = evaluate_integer(&requests[1], x, m, read, Some(&q))?;
    // Both projections already preserve base/LoRA BF16 boundaries.
    let out = gate
        .into_iter()
        .zip(up)
        .map(|(g, u)| bf(bf_silu(g) * u))
        .collect::<Vec<_>>();
    if !out.iter().all(|v| v.is_finite()) {
        return Err("fused MLP finite output".into());
    }
    Ok((out, rg + ru))
}

/// Identical pack dispatch for native validation and canister stable-memory tiles.
pub fn evaluate_with_reader<F>(
    r: &Request,
    x: &[f32],
    m: &Manifest,
    mut read: F,
) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<Vec<u8>>,
{
    if r.model != m.model || r.pack_hash != m.pack_hash {
        return Err("model mismatch".into());
    }
    if r.op == "mlp_gate_up_integer" {
        return evaluate_mlp(r, x, m, &mut read);
    }
    if matches!(
        r.op.as_str(),
        "int8_matmul" | "linear_integer_bf16" | "lora_integer"
    ) {
        return evaluate_integer(r, x, m, &mut read, None);
    }
    let mut load = |name: &str, request: &Request| {
        let t = m
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or("unknown tensor")?;
        load_weight(t, request, &mut read)
    };
    if r.op == "embed" {
        if r.dims.len() != 2
            || r.dims[0] != x.len()
            || r.dims[0] == 0
            || r.dims[0] > 128
            || r.dims[1] != 2560
        {
            return Err("embedding shape".into());
        }
        let t = m
            .tensors
            .iter()
            .find(|t| t.name == r.tensor)
            .ok_or("embedding tensor")?;
        let mut result = Vec::with_capacity(x.len() * 2560);
        let mut read = 0;
        for &id in x {
            if id < 0. || id.fract() != 0. || id as usize >= t.rows {
                return Err("token ID".into());
            }
            let mut tile = r.clone();
            tile.op = "matmul".into();
            tile.dims = vec![1, 1, 2560, id as usize];
            let (row, bytes) = load(&r.tensor, &tile)?;
            result.extend(row.into_iter().map(bf));
            read += bytes;
        }
        return Ok((result, read));
    }
    if r.op == "delta_gates" {
        if r.aux.len() != 1 {
            return Err("gate tensors".into());
        }
        let (mut w, read_a) = load(&r.tensor, r)?;
        let (b, read_b) = load(&r.aux[0], r)?;
        w.extend(b);
        return Ok((execute(r, x, &w)?, read_a + read_b));
    }
    if r.op == "lora_grouped" {
        if r.dims.len() != 5
            || r.aux.len() != 2
            || r.scalars.len() != 1
            || r.dims.iter().any(|&v| v > 262144)
        {
            return Err("grouped projection shape".into());
        }
        let (n, width, cols, start, total) =
            (r.dims[0], r.dims[1], r.dims[2], r.dims[3], r.dims[4]);
        if n == 0
            || n > 512
            || width == 0
            || total == 0
            || total > 3 * width
            || cols == 0
            || n.checked_mul(cols) != Some(x.len())
        {
            return Err("grouped projection bounds".into());
        }
        let lengths = (0..total)
            .step_by(width)
            .map(|r| n * width.min(total - r))
            .collect::<Vec<_>>();
        let padded = lengths
            .iter()
            .map(|&v| v.div_ceil(256) * 256)
            .sum::<usize>();
        if padded > MAX_FLOATS {
            return Err("grouped projection output bound".into());
        }
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[0])
            .ok_or("missing LoRA A")?;
        let b = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[1])
            .ok_or("missing LoRA B")?;
        if a.rows == 0 || a.rows > 256 || a.cols != cols || b.cols != a.rows {
            return Err("grouped LoRA shape".into());
        }
        let work = n as u64 * (total as u64 * cols as u64 + a.rows as u64 * (total + cols) as u64);
        if work > 1_350_000_000 || !r.scalars[0].is_finite() {
            return Err("grouped projection work limit".into());
        }
        let mut ar = r.clone();
        ar.op = "matmul".into();
        ar.dims = vec![n, total, cols, start];
        let (base, rb) = load(&r.tensor, &ar)?;
        ar.dims = vec![n, a.rows, cols];
        let (aw, ra) = load(&r.aux[0], &ar)?;
        ar.dims = vec![n, total, a.rows, start];
        let (bw, rz) = load(&r.aux[1], &ar)?;
        let y = lora_project_limit(
            x,
            &base,
            &aw,
            &bw,
            n,
            total,
            cols,
            a.rows,
            r.scalars[0],
            1_350_000_000,
        )?;
        let mut out = Vec::with_capacity(padded);
        for row in (0..total).step_by(width) {
            let count = width.min(total - row);
            let begin = out.len();
            for token in 0..n {
                out.extend_from_slice(&y[token * total + row..token * total + row + count]);
            }
            out.resize(begin + (n * count).div_ceil(256) * 256, 0.);
        }
        return Ok((out, rb + ra + rz));
    }
    if r.op == "lora_project" {
        if !(r.dims.len() == 3 || r.dims.len() == 4)
            || r.aux.len() != 2
            || r.scalars.len() != 1
            || r.dims.iter().any(|&d| d > 262144)
        {
            return Err("projection shape".into());
        }
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[0])
            .ok_or("missing LoRA A")?;
        let b = m
            .tensors
            .iter()
            .find(|t| t.name == r.aux[1])
            .ok_or("missing LoRA B")?;
        if a.rows == 0 || a.rows > 256 || a.cols != r.dims[2] || b.cols != a.rows {
            return Err("LoRA shape".into());
        }
        let work = (r.dims[0] as u64)
            * ((r.dims[1] as u64) * (r.dims[2] as u64)
                + (a.rows as u64) * ((r.dims[1] + r.dims[2]) as u64));
        if work > FUSED_WORK_LIMIT {
            return Err("projection work limit".into());
        }
        let (base, read_base) = load(&r.tensor, r)?;
        let mut ar = r.clone();
        ar.op = "matmul".into();
        ar.dims = vec![r.dims[0], a.rows, a.cols];
        let (aw, read_a) = load(&r.aux[0], &ar)?;
        ar.dims = vec![r.dims[0], r.dims[1], a.rows, *r.dims.get(3).unwrap_or(&0)];
        let (bw, read_b) = load(&r.aux[1], &ar)?;
        let y = lora_project(
            x,
            &base,
            &aw,
            &bw,
            r.dims[0],
            r.dims[1],
            r.dims[2],
            a.rows,
            r.scalars[0],
        )?;
        Ok((y, read_base + read_a + read_b))
    } else {
        let (w, read) = if r.tensor.is_empty() {
            (vec![], 0)
        } else {
            load(&r.tensor, r)?
        };
        Ok((execute(r, x, &w)?, read))
    }
}
#[cfg(test)]
mod full_graph_tests {
    use super::*;
    fn req(op: &str, dims: Vec<usize>, scalars: Vec<f32>) -> Request {
        Request {
            version: 1,
            model: "a".repeat(64),
            pack_hash: "b".repeat(64),
            input_hash: "c".repeat(64),
            step: 0,
            op: op.into(),
            tensor: String::new(),
            dims,
            scalars,
            aux: vec![],
            encoding: String::new(),
        }
    }
    #[test]
    fn sigmoid_matches_captured_official_bf16_values() {
        let input = [
            -2.140625,
            2.15625,
            -0.46875,
            -0.28515625,
            1.125,
            0.482421875,
        ];
        let expected = [
            0.10546875,
            0.89453125,
            0.384765625,
            0.4296875,
            0.75390625,
            0.6171875,
        ];
        for (x, y) in input.into_iter().zip(expected) {
            assert_eq!(bf_sigmoid(x), y);
        }
    }
    #[test]
    fn convolution_window_continues_across_token_queries() {
        let full = req("conv_state", vec![4, 1, 4, 0], vec![]);
        let w = [0.25, 0.5, -0.25, 1.];
        let one = execute(&full, &[0., 0., 0., 1., 2., 3., 4.], &w).unwrap();
        let half = req("conv_state", vec![2, 1, 4, 0], vec![]);
        let first = execute(&half, &[0., 0., 0., 1., 2.], &w).unwrap();
        let mut input = first[2..].to_vec();
        input.extend([3., 4.]);
        let second = execute(&half, &input, &w).unwrap();
        assert_eq!([&first[..2], &second[..2]].concat(), one[..4]);
        assert_eq!(second[2..], one[4..]);
    }
    #[test]
    fn rope_uses_partial_split_half_and_absolute_position() {
        let r = req("rope", vec![2, 6, 4, 0], vec![10000.]);
        let x = [1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.];
        let full = execute(&r, &x, &[]).unwrap();
        assert_eq!(full[..6], x[..6]);
        assert_eq!(full[10..], x[10..]);
        let next = req("rope", vec![1, 6, 4, 1], vec![10000.]);
        assert_eq!(execute(&next, &x[6..], &[]).unwrap(), full[6..]);
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    fn request() -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"bf16","tensor":"","dims":[],"scalars":[],"encoding":"bf16-exact"})).unwrap()
    }
    #[test]
    fn mixed_wire_preserves_every_bit_and_legacy_format() {
        let mut r = request();
        let values = vec![0., -0., 1., -4.25, 1.0000001, f32::from_bits(1), f32::MAX];
        let encoded = encode(&r, &values).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.encoding, "bf16-exact");
        assert_eq!(
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        r.encoding.clear();
        assert_eq!(
            decode(&encode(&r, &values).unwrap())
                .unwrap()
                .1
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn wire_bounds_and_truncation_are_rejected() {
        let r = request();
        let b = encode(&r, &vec![1.; 900_000]).unwrap();
        assert!(b.len() < 2_000_000);
        assert_eq!(decode(&b).unwrap().1.len(), 900_000);
        assert!(encode(&r, &vec![1.; 900_001]).is_err());
        assert!(encode(&r, &vec![1.0000001; 500_000]).is_err());
        let mut truncated = encode(&r, &[1., 1.0000001]).unwrap();
        truncated.truncate(truncated.len() - 33);
        let digest = Sha256::digest(&truncated);
        truncated.extend_from_slice(&digest);
        assert!(decode(&truncated).is_err());
        assert!(encode(&r, &[f32::NAN]).is_err());
    }
}

#[cfg(test)]
mod int8_wire_tests {
    use super::*;
    fn request(op: &str, dims: Vec<usize>) -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":op,"tensor":"","dims":dims,"scalars":[],"encoding":"int8-block256-v1"})).unwrap()
    }
    #[test]
    fn delta_tail_and_token_ids_are_never_quantized() {
        let r = request("delta_bf16", vec![2, 128, 128]);
        for (count, prefix) in [(2 * 386 + 16384, 2 * 384), (2 * 128 + 16384, 2 * 128)] {
            let values = (0..count).map(|i| i as f32 * 0.0000137).collect::<Vec<_>>();
            let (_, got) = decode(&encode(&r, &values).unwrap()).unwrap();
            assert_eq!(&got[prefix..], &values[prefix..]);
        }
        let r = request("embed", vec![2, 2560]);
        assert_eq!(
            decode(&encode(&r, &[248319., 1.]).unwrap()).unwrap().1,
            vec![248319., 1.]
        );
    }
    #[test]
    fn block_codec_rounds_ties_even_and_handles_partial_block() {
        let r = request("bf16", vec![]);
        let mut values = vec![0.; 257];
        values[..5].copy_from_slice(&[127., 0.5, 1.5, -0.5, -1.5]);
        values[256] = 3.;
        let (_, got) = decode(&encode(&r, &values).unwrap()).unwrap();
        assert_eq!(&got[..5], &[127., 0., 2., 0., -2.]);
        assert_eq!(got[256], 3.);
        assert!(encode(&r, &[f32::INFINITY]).is_err());
    }
    #[test]
    fn malformed_scale_and_reserved_int8_code_are_rejected() {
        let r = request("bf16", vec![]);
        let encoded = encode(&r, &[1.]).unwrap();
        let n = u32::from_le_bytes(encoded[..4].try_into().unwrap()) as usize;
        for (offset, bytes) in [(4 + n + 8, vec![0; 4]), (4 + n + 12, vec![128])] {
            let mut b = encoded[..encoded.len() - 32].to_vec();
            b[offset..offset + bytes.len()].copy_from_slice(&bytes);
            let digest = profile::measure("wire_encode_sha256", || Sha256::digest(&b));
            b.extend_from_slice(&digest);
            assert!(decode(&b).is_err());
        }
    }
}

#[cfg(test)]
mod flat_pair_contract_tests {
    use super::*;
    #[test]
    fn only_bounded_binary_element_ops_accept_large_flat_lengths() {
        let mut r: Request = serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"add_bf16","tensor":"","dims":[450000],"scalars":[],"encoding":"bf16-exact"})).unwrap();
        let x = vec![0.; 900000];
        assert_eq!(execute(&r, &x, &[]).unwrap().len(), 450000);
        assert!(execute(&r, &x[..899999], &[]).is_err());
        r.dims[0] = 450001;
        assert!(execute(&r, &vec![0.; 900002], &[]).is_err());
        r.op = "bf16".into();
        r.dims[0] = 450000;
        assert!(execute(&r, &x, &[]).is_err());
    }
}

#[cfg(test)]
mod add_norm_tests {
    use super::*;
    #[test]
    fn fusion_preserves_add_rounding_before_rms() {
        let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"add_norm_bf16","tensor":"","dims":[3,7],"scalars":[1e-6],"encoding":"bf16-exact"})).unwrap();
        let x: Vec<_> = (0..42).map(|i| bf((i as f32 - 20.) * 0.00317)).collect();
        let weights = vec![1.; 7];
        let got = execute(&r, &x, &weights).unwrap();
        r.op = "add_bf16".into();
        r.dims = vec![21];
        r.scalars.clear();
        let added = execute(&r, &x, &[]).unwrap();
        r.op = "rms_bf16".into();
        r.dims = vec![3, 7];
        r.scalars = vec![1e-6];
        let norm = execute(&r, &added, &weights).unwrap();
        let expected: Vec<_> = added.into_iter().chain(norm).collect();
        assert_eq!(got, expected);
        r.op = "add_norm_bf16".into();
        assert!(execute(&r, &x[..41], &weights).is_err());
        r.scalars = vec![-1.];
        assert!(execute(&r, &x, &weights).is_err());
    }
}
#[cfg(test)]
mod fused_mlp_tests {
    use super::*;
    #[test]
    fn fused_gate_up_preserves_two_projection_rounding_and_rejects_metadata() {
        let model = "a".repeat(64);
        let hash = "b".repeat(64);
        let mut pack = Vec::new();
        let mut tensors = Vec::new();
        for (index, kind) in ["gate", "up"].iter().enumerate() {
            let prefix = format!("model.language_model.layers.0.mlp.{kind}_proj");
            let offset = pack.len() as u64;
            pack.extend((0..8 * 256).map(|i| ((i % 7) as i8 - 3) as u8));
            for _ in 0..8 {
                pack.extend_from_slice(&0.01f32.to_le_bytes());
            }
            tensors.push(Tensor {
                name: format!("{prefix}.weight"),
                offset,
                rows: 8,
                cols: 256,
                dtype: "int8".into(),
                bytes: 2080,
            });
            let offset = pack.len() as u64;
            for i in 0..256 {
                pack.extend_from_slice(&((i % 11) as f32 * 0.001).to_le_bytes());
            }
            tensors.push(Tensor {
                name: format!("{prefix}.lora_A.weight"),
                offset,
                rows: 1,
                cols: 256,
                dtype: "f32".into(),
                bytes: 1024,
            });
            let offset = pack.len() as u64;
            for i in 0..8 {
                pack.extend_from_slice(&((i + index) as f32 * 0.003).to_le_bytes());
            }
            tensors.push(Tensor {
                name: format!("{prefix}.lora_B.weight"),
                offset,
                rows: 8,
                cols: 1,
                dtype: "f32".into(),
                bytes: 32,
            });
        }
        let manifest = Manifest {
            version: 1,
            model: model.clone(),
            pack_hash: hash.clone(),
            bytes: pack.len() as u64,
            tensors,
        };
        let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":model,"pack_hash":hash,"input_hash":"c".repeat(64),"step":0,"op":"mlp_gate_up_integer","tensor":"model.language_model.layers.0.mlp.gate_proj.weight","dims":[2,8,256,0],"scalars":[2.],"aux":["model.language_model.layers.0.mlp.up_proj.weight"]})).unwrap();
        let x: Vec<_> = (0..512)
            .map(|i| bf((i % 17) as f32 * 0.007 - 0.03))
            .collect();
        let read = |offset: u64, len: usize| {
            pack.get(offset as usize..offset as usize + len)
                .map(|v| v.to_vec())
                .ok_or("read bounds".into())
        };
        let (got, _) = evaluate_with_reader(&r, &x, &manifest, read).unwrap();
        let mut outputs = Vec::new();
        for name in [&r.tensor, &r.aux[0]] {
            let mut request = r.clone();
            request.op = "lora_integer".into();
            request.tensor = name.clone();
            let prefix = name.strip_suffix(".weight").unwrap();
            request.aux = vec![
                format!("{prefix}.lora_A.weight"),
                format!("{prefix}.lora_B.weight"),
            ];
            outputs.extend(
                evaluate_with_reader(&request, &x, &manifest, read)
                    .unwrap()
                    .0,
            );
        }
        let mut sw = r.clone();
        sw.op = "swiglu_bf16".into();
        sw.dims = vec![16];
        sw.scalars.clear();
        assert_eq!(got, execute(&sw, &outputs, &[]).unwrap());
        r.aux.clear();
        assert!(evaluate_with_reader(&r, &x, &manifest, |_, _| panic!(
            "invalid metadata must not read weights"
        ))
        .is_err());
        r.aux = vec!["model.language_model.layers.1.mlp.up_proj.weight".into()];
        assert!(evaluate_with_reader(&r, &x, &manifest, read).is_err());
        r.aux = vec!["model.language_model.layers.0.mlp.up_proj.weight".into()];
        r.dims[1] = 9;
        assert!(evaluate_with_reader(&r, &x, &manifest, read).is_err());
    }
}
