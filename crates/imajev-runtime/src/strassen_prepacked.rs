//! Exact block256 Strassen trial with precomputed INT8 operands and sparse carries.
//! Diagnostic feature only: no extra quantization and no persistent query state.
use crate::{bf, int8_kernel, Manifest, Request, Result};
#[cfg(feature = "experimental-strassen-wide")]
type StrassenWeight = i16;
#[cfg(not(feature = "experimental-strassen-wide"))]
type StrassenWeight = i8;

struct Corrections {
    offsets: Vec<usize>,
    entries: Vec<(usize, i32)>,
}
/// Fixed exact I16 coefficients. Construction verifies them against the base
/// INT8 tensor; inference never restores carries or scans these coefficients.
#[cfg(feature = "experimental-strassen-prepared")]
pub struct PreparedStrassen {
    model: String,
    pack_hash: String,
    source: String,
    rows: usize,
    cols: usize,
    weights: Vec<i16>,
    scales: Vec<f32>,
}
#[cfg(feature = "experimental-strassen-prepared")]
impl PreparedStrassen {
    fn request(m: &Manifest, source: &str) -> Result<Request> {
        let t = m.tensors.iter().find(|t| t.name == source).ok_or("prepared Strassen source")?;
        Ok(Request { version: 1, model: m.model.clone(), pack_hash: m.pack_hash.clone(), input_hash: "0".repeat(64), step: 0,
            op: "linear_strassen_bf16".into(), tensor: source.into(), dims: vec![1, t.rows, t.cols, 0], scalars: vec![],
            aux: vec![format!("{source}.strassen_i8"), format!("{source}.strassen_corrections")], encoding: "bf16-exact".into() })
    }
    pub fn required_bytes(m: &Manifest, source: &str) -> Result<u64> {
        let r = Self::request(m, source)?;
        let (original, derived, _, _) = layout(&r, r.dims[2], m)?;
        derived.bytes.checked_mul(2).and_then(|v| v.checked_add((original.rows * 4) as u64)).ok_or("prepared Strassen size".into())
    }
    pub fn source(&self) -> &str { &self.source }
    pub fn bytes(&self) -> u64 { (self.weights.len() * 2 + self.scales.len() * 4) as u64 }
    pub fn prepare<F, B>(m: &Manifest, source: &str, mut read: F) -> Result<Self>
    where F: FnMut(u64, usize) -> Result<B>, B: AsRef<[u8]>,
    {
        let r = Self::request(m, source)?;
        let (original, derived, carries, records) = layout(&r, r.dims[2], m)?;
        if Self::required_bytes(m, source)? > 128 * 1024 * 1024 { return Err("prepared Strassen tensor limit".into()); }
        let raw = read(original.offset, original.bytes as usize)?;
        let wrapped = read(derived.offset, derived.bytes as usize)?;
        let csr = read(carries.offset, carries.bytes as usize)?;
        let (raw, wrapped, csr) = (raw.as_ref(), wrapped.as_ref(), csr.as_ref());
        if raw.len() as u64 != original.bytes || wrapped.len() as u64 != derived.bytes || csr.len() as u64 != carries.bytes {
            return Err("prepared Strassen read size".into());
        }
        let corrections = Corrections::parse(csr, records)?;
        let wrapped: Vec<_> = wrapped.iter().map(|b| *b as i8).collect();
        let weights = widen(&wrapped, original.rows, original.cols, &corrections)?;
        let cols = original.cols;
        let stride = cols / 2;
        for product in 0..7 {
            for pair in 0..original.rows / 2 {
                for block in 0..cols / 256 {
                    for k in 0..128 {
                        let p = pair * 2 * cols + block * 256 + k;
                        let (b11, b21, b12, b22) = (raw[p] as i8 as i16, raw[p + 128] as i8 as i16,
                            raw[p + cols] as i8 as i16, raw[p + cols + 128] as i8 as i16);
                        let expected = [b11 + b22, b11, b12 - b22, b21 - b11, b22, b11 + b12, b21 + b22][product];
                        if weights[(product * (original.rows / 2) + pair) * stride + block * 128 + k] != expected {
                            return Err("prepared Strassen coefficients differ from base".into());
                        }
                    }
                }
            }
        }
        let scales: Vec<_> = raw[original.rows * cols..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        if !scales.iter().all(|v| v.is_finite() && *v > 0.) { return Err("prepared Strassen scales".into()); }
        Ok(Self { model: m.model.clone(), pack_hash: m.pack_hash.clone(), source: source.into(), rows: original.rows, cols, weights, scales })
    }
    pub fn evaluate(&self, r: &Request, x: &[f32]) -> Result<Vec<f32>> {
        if r.op != "linear_strassen_bf16" || r.model != self.model || r.pack_hash != self.pack_hash || r.tensor != self.source
            || r.dims.len() != 4 || r.dims[1] != self.rows || r.dims[2] != self.cols || r.dims[3] != 0
            || r.dims[0] == 0 || r.dims[0] > 512 || r.dims[0].checked_mul(self.cols) != Some(x.len())
            || r.dims[0].checked_mul(self.rows).is_none_or(|v| v > crate::MAX_FLOATS)
            || r.dims[0] as u64 * self.rows as u64 * self.cols as u64 > 4_000_000_000
            || !r.scalars.is_empty() || !crate::lossless_encoding(&r.encoding)
            || r.aux != [format!("{}.strassen_i8", self.source), format!("{}.strassen_corrections", self.source)] {
            return Err("prepared Strassen request".into());
        }
        let q = int8_kernel::quantize_rows(x, r.dims[0], self.cols)?;
        let unused = Corrections { offsets: vec![], entries: vec![] };
        #[cfg(target_arch = "wasm32")]
        let out = unsafe { project_simd(&q, &self.weights, &self.scales, self.rows, &unused) };
        #[cfg(not(target_arch = "wasm32"))]
        let out = project_scalar(&q, &self.weights, &self.scales, self.rows, &unused);
        if !out.iter().all(|v| v.is_finite()) { return Err("prepared Strassen output".into()); }
        Ok(out.into_iter().map(bf).collect())
    }
}
impl Corrections {
    fn parse(b: &[u8], records: usize) -> Result<Self> {
        if b.len() < 4 || u32::from_le_bytes(b[..4].try_into().unwrap()) as usize != records {
            return Err("Strassen correction records".into());
        }
        let header = records
            .checked_add(1)
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| v.checked_add(4))
            .ok_or("Strassen correction size")?;
        if b.len() < header || (b.len() - header) % 2 != 0 {
            return Err("Strassen correction length".into());
        }
        let offsets: Vec<usize> = b[4..header]
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize)
            .collect();
        let count = (b.len() - header) / 2;
        if offsets[0] != 0 || offsets[records] != count || offsets.windows(2).any(|v| v[0] > v[1]) {
            return Err("Strassen correction offsets".into());
        }
        let mut entries = Vec::with_capacity(count);
        for v in b[header..].chunks_exact(2) {
            let sign = v[1] as i8;
            if v[0] >= 128 || !matches!(sign, -1 | 1) {
                return Err("Strassen correction entry".into());
            }
            entries.push((v[0] as usize, sign as i32));
        }
        for p in offsets.windows(2) {
            if entries[p[0]..p[1]].windows(2).any(|v| v[0].0 >= v[1].0) {
                return Err("Strassen correction order".into());
            }
        }
        Ok(Self { offsets, entries })
    }
}
fn layout<'a>(r: &Request, input_len: usize, m: &'a Manifest) -> Result<(&'a crate::Tensor, &'a crate::Tensor, &'a crate::Tensor, usize)> {
    if r.dims.len() != 4 || r.aux.len() != 2 || !r.scalars.is_empty() || !crate::lossless_encoding(&r.encoding)
    {
        return Err("Strassen metadata".into());
    }
    let (n, rows, cols, start) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3]);
    if n == 0
        || n > 512
        || rows == 0
        || rows % 8 != 0
        || cols == 0
        || cols % 256 != 0
        || start != 0
        || n.checked_mul(cols) != Some(input_len)
        || n.checked_mul(rows).is_none_or(|v| v > crate::MAX_FLOATS)
        || rows.checked_mul(cols).is_none_or(|v| v > 30_000_000)
        || n as u64 * rows as u64 * cols as u64 > 4_000_000_000
    {
        return Err("Strassen shape/work bounds".into());
    }
    let original = m
        .tensors
        .iter()
        .find(|t| t.name == r.tensor)
        .ok_or("Strassen source tensor")?;
    let derived = m
        .tensors
        .iter()
        .find(|t| t.name == r.aux[0])
        .ok_or("Strassen derived tensor")?;
    let carries = m
        .tensors
        .iter()
        .find(|t| t.name == r.aux[1])
        .ok_or("Strassen carries tensor")?;
    let half = rows / 2 * (cols / 2);
    let records = 7 * (rows / 2) * (cols / 256);
    if original.dtype != "int8"
        || original.rows != rows
        || original.cols != cols
        || original.bytes != (rows * cols + rows * 4) as u64
        || derived.name != format!("{}.strassen_i8", r.tensor)
        || carries.name != format!("{}.strassen_corrections", r.tensor)
        || derived.dtype != "strassen-i8-v1"
        || derived.rows != rows / 2
        || derived.cols != cols / 2
        || derived.bytes != (7 * half) as u64
        || carries.dtype != "strassen-csr-v1"
        || carries.rows != rows / 2
        || carries.cols != cols / 2
        || carries.bytes > 128 * 1024 * 1024
    {
        return Err("Strassen tensor layout".into());
    }
    for t in [original, derived, carries] {
        if t.offset.checked_add(t.bytes).is_none_or(|v| v > m.bytes) {
            return Err("Strassen tensor range".into());
        }
    }
    Ok((original, derived, carries, records))
}

pub fn evaluate<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: AsRef<[u8]>,
{
    let (original, derived, carries, records) = layout(r, x.len(), m)?;
    let (n, rows, cols) = (r.dims[0], r.dims[1], r.dims[2]);
    let w = read(derived.offset, derived.bytes as usize)?;
    if w.as_ref().len() != derived.bytes as usize {
        return Err("Strassen weight read".into());
    }
    let b = read(carries.offset, carries.bytes as usize)?;
    let corrections = Corrections::parse(b.as_ref(), records)?;
    let scale_bytes = read(original.offset + (rows * cols) as u64, rows * 4)?;
    if scale_bytes.as_ref().len() != rows * 4 {
        return Err("Strassen scale read".into());
    }
    let sw: Vec<f32> = scale_bytes.as_ref()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if !sw.iter().all(|v| v.is_finite() && *v > 0.) {
        return Err("Strassen scales".into());
    }
    let q = int8_kernel::quantize_rows(x, n, cols)?;
    let wrapped: Vec<i8> = w.as_ref().iter().map(|v| *v as i8).collect();
    #[cfg(feature = "experimental-strassen-wide")]
    let weights = crate::profile::measure("strassen_widen", || widen(&wrapped, rows, cols, &corrections))?;
    #[cfg(not(feature = "experimental-strassen-wide"))]
    let weights = wrapped;
    #[cfg(target_arch = "wasm32")]
    let out = unsafe { project_simd(&q, &weights, &sw, rows, &corrections) };
    #[cfg(not(target_arch = "wasm32"))]
    let out = project_scalar(&q, &weights, &sw, rows, &corrections);
    if !out.iter().all(|v| v.is_finite()) {
        return Err("Strassen nonfinite output".into());
    }
    Ok((
        out.into_iter().map(bf).collect(),
        derived.bytes + carries.bytes + (rows * 4) as u64,
    ))
}
fn operands(q: &int8_kernel::QuantizedRows) -> Vec<i16> {
    let tokens = q.rows().div_ceil(8) * 8;
    let stride = q.cols() / 2;
    let mut a = vec![0i16; 7 * (tokens / 2) * stride];
    for pair in 0..tokens / 2 {
        for block in 0..q.cols() / 256 {
            for k in 0..128 {
                let p = pair * 2 * q.cols() + block * 256 + k;
                let (a11, a12, a21, a22) = (
                    q.values()[p],
                    q.values()[p + 128],
                    q.values()[p + q.cols()],
                    q.values()[p + q.cols() + 128],
                );
                let v = [
                    a11 + a22,
                    a21 + a22,
                    a11,
                    a22,
                    a11 + a12,
                    a21 - a11,
                    a12 - a22,
                ];
                for m in 0..7 {
                    a[m * (tokens / 2) * stride + pair * stride + block * 128 + k] = v[m];
                }
            }
        }
    }
    a
}
/// Restore the original integer operands once, before token/output tiles.
/// The immutable pack remains INT8 plus sparse signed carries. This is an exact
/// query-local representation, not another weight quantization.
#[cfg(feature = "experimental-strassen-wide")]
fn widen(w: &[i8], rows: usize, cols: usize, c: &Corrections) -> Result<Vec<i16>> {
    let stride = cols / 2;
    let blocks = cols / 256;
    let mut wide: Vec<i16> = w.iter().map(|v| *v as i16).collect();
    for m in 0..7 {
        for j in 0..rows / 2 {
            for block in 0..blocks {
                let record = (m * (rows / 2) + j) * blocks + block;
                let start = (m * (rows / 2) + j) * stride + block * 128;
                for &(k, sign) in &c.entries[c.offsets[record]..c.offsets[record + 1]] {
                    let value = wide[start + k] as i32 + sign * 256;
                    if !(-256..=256).contains(&value) {
                        return Err("Strassen widened operand range".into());
                    }
                    wide[start + k] = value as i16;
                }
            }
        }
    }
    Ok(wide)
}
#[cfg(not(target_arch = "wasm32"))]
fn project_scalar(
    q: &int8_kernel::QuantizedRows,
    w: &[StrassenWeight],
    sw: &[f32],
    rows: usize,
    c: &Corrections,
) -> Vec<f32> {
    let tokens = q.rows().div_ceil(8) * 8;
    let stride = q.cols() / 2;
    let blocks = q.cols() / 256;
    let a = operands(q);
    let mut out = vec![0.; q.rows() * rows];
    for pair in 0..tokens / 2 {
        for j in 0..rows / 2 {
            for block in 0..blocks {
                let mut p = [0i32; 7];
                for m in 0..7 {
                    let ai = m * (tokens / 2) * stride + pair * stride + block * 128;
                    let wi = m * (rows / 2) * stride + j * stride + block * 128;
                    for k in 0..128 {
                        p[m] += a[ai + k] as i32 * w[wi + k] as i32;
                    }
                    let record = (m * (rows / 2) + j) * blocks + block;
                    #[cfg(not(feature = "experimental-strassen-wide"))]
                    for &(k, sign) in &c.entries[c.offsets[record]..c.offsets[record + 1]] {
                        p[m] += a[ai + k] as i32 * sign * 256;
                    }
                }
                let d = [
                    [p[0] + p[3] - p[4] + p[6], p[2] + p[4]],
                    [p[1] + p[3], p[0] - p[1] + p[2] + p[5]],
                ];
                for ti in 0..2 {
                    let t = pair * 2 + ti;
                    if t < q.rows() {
                        for ri in 0..2 {
                            let r = j * 2 + ri;
                            out[t * rows + r] +=
                                (d[ti][ri] as f32 * q.scales()[t * blocks + block]) * sw[r];
                        }
                    }
                }
            }
        }
    }
    out
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn project_simd(
    q: &int8_kernel::QuantizedRows,
    w: &[StrassenWeight],
    sw: &[f32],
    rows: usize,
    c: &Corrections,
) -> Vec<f32> {
    let a = operands(q);
    let tokens = q.rows().div_ceil(8) * 8;
    let mut out = vec![0.; q.rows() * rows];
    for r in (0..rows).step_by(8) {
        let mut t = 0;
        while t < q.rows() {
            macro_rules! tile {
                ($n:literal) => {{
                    project_tile::<$n>(&a, w, q, sw, rows, c, t, r, &mut out);
                    t += 2 * $n;
                }};
            }
            if t + 64 <= tokens {
                tile!(32);
            } else if t + 32 <= tokens {
                tile!(16);
            } else if t + 16 <= tokens {
                tile!(8);
            } else {
                tile!(4);
            }
        }
    }
    out
}
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn project_tile<const R: usize>(
    a: &[i16],
    w: &[StrassenWeight],
    q: &int8_kernel::QuantizedRows,
    sw: &[f32],
    rows: usize,
    c: &Corrections,
    t: usize,
    r: usize,
    out: &mut [f32],
) {
    use core::arch::wasm32::*;
    let tokens = q.rows().div_ceil(8) * 8;
    let stride = q.cols() / 2;
    let blocks = q.cols() / 256;
    let zero = f32x4_splat(0.);
    let mut sums = [[[zero; 2]; 2]; R];
    let lo = v128_load(sw.as_ptr().add(r).cast());
    let hi = v128_load(sw.as_ptr().add(r + 4).cast());
    let even = i32x4_shuffle::<0, 2, 4, 6>(lo, hi);
    let odd = i32x4_shuffle::<1, 3, 5, 7>(lo, hi);
    for block in 0..blocks {
        let mut products: [[[i32; 4]; R]; 7] = core::array::from_fn(|m| {
            dot_tile::<R>(
                a.as_ptr()
                    .add(m * (tokens / 2) * stride + t / 2 * stride + block * 128),
                w.as_ptr()
                    .add(m * (rows / 2) * stride + r / 2 * stride + block * 128),
                stride,
            )
        });
        #[cfg(not(feature = "experimental-strassen-wide"))]
        for m in 0..7 {
            for j in 0..4 {
                let record = (m * (rows / 2) + r / 2 + j) * blocks + block;
                let slice = &c.entries[c.offsets[record]..c.offsets[record + 1]];
                for &(k, sign) in slice {
                    let ai = m * (tokens / 2) * stride + t / 2 * stride + block * 128 + k;
                    for i in 0..R {
                        products[m][i][j] += *a.get_unchecked(ai + i * stride) as i32 * sign * 256;
                    }
                }
            }
        }
        for i in 0..R {
            let p0 = v128_load(products[0][i].as_ptr().cast());
            let p1 = v128_load(products[1][i].as_ptr().cast());
            let p2 = v128_load(products[2][i].as_ptr().cast());
            let p3 = v128_load(products[3][i].as_ptr().cast());
            let p4 = v128_load(products[4][i].as_ptr().cast());
            let p5 = v128_load(products[5][i].as_ptr().cast());
            let p6 = v128_load(products[6][i].as_ptr().cast());
            let d = [
                [
                    i32x4_add(i32x4_sub(i32x4_add(p0, p3), p4), p6),
                    i32x4_add(p2, p4),
                ],
                [
                    i32x4_add(p1, p3),
                    i32x4_add(i32x4_add(i32x4_sub(p0, p1), p2), p5),
                ],
            ];
            for ti in 0..2 {
                let sx = f32x4_splat(*q.scales().get_unchecked((t + i * 2 + ti) * blocks + block));
                for ri in 0..2 {
                    let weight = if ri == 0 { even } else { odd };
                    let value = f32x4_mul(f32x4_mul(f32x4_convert_i32x4(d[ti][ri]), sx), weight);
                    sums[i][ti][ri] = f32x4_add(sums[i][ti][ri], value);
                }
            }
        }
    }
    for i in 0..R {
        for ti in 0..2 {
            let token = t + i * 2 + ti;
            if token < q.rows() {
                let even = sums[i][ti][0];
                let odd = sums[i][ti][1];
                let lo = i32x4_shuffle::<0, 4, 1, 5>(even, odd);
                let hi = i32x4_shuffle::<2, 6, 3, 7>(even, odd);
                let p = out.as_mut_ptr().add(token * rows + r);
                v128_store(p.cast(), lo);
                v128_store(p.add(4).cast(), hi);
            }
        }
    }
}

#[cfg(not(feature = "experimental-strassen-packed-reduction"))]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_tile<const R: usize>(q: *const i16, w: *const StrassenWeight, stride: usize) -> [[i32; 4]; R] {
    use core::arch::wasm32::*;
    let weights: [[v128; 16]; 4] = core::array::from_fn(|j| {
        core::array::from_fn(|k| {
            #[cfg(feature = "experimental-strassen-wide")]
            { v128_load(w.add(j * stride + k * 8).cast()) }
            #[cfg(not(feature = "experimental-strassen-wide"))]
            { i16x8_extend_low_i8x16(v128_load64_zero(w.add(j * stride + k * 8).cast())) }
        })
    });
    let mut out = [[0i32; 4]; R];
    macro_rules! row {
        ($i:literal) => {
            if R > $i {
                let input: [v128; 16] =
                    core::array::from_fn(|k| v128_load(q.add($i * stride + k * 8).cast()));
                {
                    let v = i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[0], weights[0][0]),
                                    i32x4_dot_i16x8(input[1], weights[0][1]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[2], weights[0][2]),
                                    i32x4_dot_i16x8(input[3], weights[0][3]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[4], weights[0][4]),
                                    i32x4_dot_i16x8(input[5], weights[0][5]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[6], weights[0][6]),
                                    i32x4_dot_i16x8(input[7], weights[0][7]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[8], weights[0][8]),
                                    i32x4_dot_i16x8(input[9], weights[0][9]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[10], weights[0][10]),
                                    i32x4_dot_i16x8(input[11], weights[0][11]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[12], weights[0][12]),
                                    i32x4_dot_i16x8(input[13], weights[0][13]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[14], weights[0][14]),
                                    i32x4_dot_i16x8(input[15], weights[0][15]),
                                ),
                            ),
                        ),
                    );
                    let h = i32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
                    let sum = i32x4_add(h, i32x4_shuffle::<1, 0, 3, 2>(h, h));
                    out[$i][0] = i32x4_extract_lane::<0>(sum);
                }
                {
                    let v = i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[0], weights[1][0]),
                                    i32x4_dot_i16x8(input[1], weights[1][1]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[2], weights[1][2]),
                                    i32x4_dot_i16x8(input[3], weights[1][3]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[4], weights[1][4]),
                                    i32x4_dot_i16x8(input[5], weights[1][5]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[6], weights[1][6]),
                                    i32x4_dot_i16x8(input[7], weights[1][7]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[8], weights[1][8]),
                                    i32x4_dot_i16x8(input[9], weights[1][9]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[10], weights[1][10]),
                                    i32x4_dot_i16x8(input[11], weights[1][11]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[12], weights[1][12]),
                                    i32x4_dot_i16x8(input[13], weights[1][13]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[14], weights[1][14]),
                                    i32x4_dot_i16x8(input[15], weights[1][15]),
                                ),
                            ),
                        ),
                    );
                    let h = i32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
                    let sum = i32x4_add(h, i32x4_shuffle::<1, 0, 3, 2>(h, h));
                    out[$i][1] = i32x4_extract_lane::<0>(sum);
                }
                {
                    let v = i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[0], weights[2][0]),
                                    i32x4_dot_i16x8(input[1], weights[2][1]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[2], weights[2][2]),
                                    i32x4_dot_i16x8(input[3], weights[2][3]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[4], weights[2][4]),
                                    i32x4_dot_i16x8(input[5], weights[2][5]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[6], weights[2][6]),
                                    i32x4_dot_i16x8(input[7], weights[2][7]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[8], weights[2][8]),
                                    i32x4_dot_i16x8(input[9], weights[2][9]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[10], weights[2][10]),
                                    i32x4_dot_i16x8(input[11], weights[2][11]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[12], weights[2][12]),
                                    i32x4_dot_i16x8(input[13], weights[2][13]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[14], weights[2][14]),
                                    i32x4_dot_i16x8(input[15], weights[2][15]),
                                ),
                            ),
                        ),
                    );
                    let h = i32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
                    let sum = i32x4_add(h, i32x4_shuffle::<1, 0, 3, 2>(h, h));
                    out[$i][2] = i32x4_extract_lane::<0>(sum);
                }
                {
                    let v = i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[0], weights[3][0]),
                                    i32x4_dot_i16x8(input[1], weights[3][1]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[2], weights[3][2]),
                                    i32x4_dot_i16x8(input[3], weights[3][3]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[4], weights[3][4]),
                                    i32x4_dot_i16x8(input[5], weights[3][5]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[6], weights[3][6]),
                                    i32x4_dot_i16x8(input[7], weights[3][7]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[8], weights[3][8]),
                                    i32x4_dot_i16x8(input[9], weights[3][9]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[10], weights[3][10]),
                                    i32x4_dot_i16x8(input[11], weights[3][11]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8(input[12], weights[3][12]),
                                    i32x4_dot_i16x8(input[13], weights[3][13]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8(input[14], weights[3][14]),
                                    i32x4_dot_i16x8(input[15], weights[3][15]),
                                ),
                            ),
                        ),
                    );
                    let h = i32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
                    let sum = i32x4_add(h, i32x4_shuffle::<1, 0, 3, 2>(h, h));
                    out[$i][3] = i32x4_extract_lane::<0>(sum);
                }
            }
        };
    };
    row!(0);
    row!(1);
    row!(2);
    row!(3);
    row!(4);
    row!(5);
    row!(6);
    row!(7);
    row!(8);
    row!(9);
    row!(10);
    row!(11);
    row!(12);
    row!(13);
    row!(14);
    row!(15);
    row!(16);
    row!(17);
    row!(18);
    row!(19);
    row!(20);
    row!(21);
    row!(22);
    row!(23);
    row!(24);
    row!(25);
    row!(26);
    row!(27);
    row!(28);
    row!(29);
    row!(30);
    row!(31);
    out
}

#[cfg(all(target_arch = "wasm32", feature = "experimental-strassen-packed-reduction"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_tile<const R: usize>(q: *const i16, w: *const StrassenWeight, stride: usize) -> [[i32; 4]; R] {
    use core::arch::wasm32::*;
    const { assert!(R <= 32); }
    // Validate the row arithmetic once and share it across all 16 loads.
    // The caller's validated shape keeps both pointer offsets in one allocation.
    let qp: [*const i16; R] = core::array::from_fn(|i| q.add(i * stride));
    let wp: [*const StrassenWeight; 4] = core::array::from_fn(|j| w.add(j * stride));
    let weights: [[v128; 16]; 4] = core::array::from_fn(|j| core::array::from_fn(|k| v128_load(wp[j].add(k * 8).cast())));
    let mut out = [[0i32; 4]; R];
    macro_rules! dot { ($input:ident, $j:literal) => {i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[0], weights[$j][0]), i32x4_dot_i16x8($input[1], weights[$j][1])), i32x4_add(i32x4_dot_i16x8($input[2], weights[$j][2]), i32x4_dot_i16x8($input[3], weights[$j][3]))), i32x4_add(i32x4_add(i32x4_dot_i16x8($input[4], weights[$j][4]), i32x4_dot_i16x8($input[5], weights[$j][5])), i32x4_add(i32x4_dot_i16x8($input[6], weights[$j][6]), i32x4_dot_i16x8($input[7], weights[$j][7])))), i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[8], weights[$j][8]), i32x4_dot_i16x8($input[9], weights[$j][9])), i32x4_add(i32x4_dot_i16x8($input[10], weights[$j][10]), i32x4_dot_i16x8($input[11], weights[$j][11]))), i32x4_add(i32x4_add(i32x4_dot_i16x8($input[12], weights[$j][12]), i32x4_dot_i16x8($input[13], weights[$j][13])), i32x4_add(i32x4_dot_i16x8($input[14], weights[$j][14]), i32x4_dot_i16x8($input[15], weights[$j][15])))))}; }
    macro_rules! row { ($i:literal) => { if R > $i {
        let input: [v128; 16] = core::array::from_fn(|k| v128_load(qp[$i].add(k * 8).cast()));
        let a0 = dot!(input, 0); let a1 = dot!(input, 1);
        let a2 = dot!(input, 2); let a3 = dot!(input, 3);
        let a = i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1), i32x4_shuffle::<2,3,6,7>(a0,a1));
        let b = i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3), i32x4_shuffle::<2,3,6,7>(a2,a3));
        let total = i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b), i32x4_shuffle::<1,3,5,7>(a,b));
        v128_store(out[$i].as_mut_ptr().cast(), total);
    } }; }
    row!(0);
    row!(1);
    row!(2);
    row!(3);
    row!(4);
    row!(5);
    row!(6);
    row!(7);
    row!(8);
    row!(9);
    row!(10);
    row!(11);
    row!(12);
    row!(13);
    row!(14);
    row!(15);
    row!(16);
    row!(17);
    row!(18);
    row!(19);
    row!(20);
    row!(21);
    row!(22);
    row!(23);
    row!(24);
    row!(25);
    row!(26);
    row!(27);
    row!(28);
    row!(29);
    row!(30);
    row!(31);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(n: usize) -> (Request, Manifest, Vec<u8>, Vec<f32>, Vec<i8>, Vec<f32>) {
        let (rows, cols) = (8, 512);
        let w: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 71 % 256) as u8) as i8)
            .collect();
        let sw: Vec<f32> = (0..rows).map(|r| 0.011 + r as f32 * 0.003).collect();
        let x: Vec<f32> = (0..n * cols)
            .map(|i| (i as f32 % 83. - 41.) / 17.)
            .collect();
        let mut data = w.iter().map(|v| *v as u8).collect::<Vec<_>>();
        for v in &sw {
            data.extend(v.to_le_bytes());
        }
        let tensor = crate::Tensor {
            name: "base".into(),
            offset: 0,
            rows,
            cols,
            dtype: "int8".into(),
            bytes: data.len() as u64,
        };
        let mut derived = Vec::new();
        let mut offsets = vec![0u32];
        let mut entries = vec![];
        for m in 0..7 {
            for r in 0..rows / 2 {
                for block in 0..cols / 256 {
                    for k in 0..128 {
                        let p = r * 2 * cols + block * 256 + k;
                        let (b11, b21, b12, b22) = (
                            w[p] as i16,
                            w[p + 128] as i16,
                            w[p + cols] as i16,
                            w[p + cols + 128] as i16,
                        );
                        let value = [
                            b11 + b22,
                            b11,
                            b12 - b22,
                            b21 - b11,
                            b22,
                            b11 + b12,
                            b21 + b22,
                        ][m];
                        let byte = value as i8;
                        derived.push(byte as u8);
                        let flag = (value - byte as i16) / 256;
                        if flag != 0 {
                            entries.extend([k as u8, flag as i8 as u8]);
                        }
                    }
                    offsets.push((entries.len() / 2) as u32);
                }
            }
        }
        let operand = crate::Tensor {
            name: "base.strassen_i8".into(),
            offset: data.len() as u64,
            rows: rows / 2,
            cols: cols / 2,
            dtype: "strassen-i8-v1".into(),
            bytes: derived.len() as u64,
        };
        data.extend(derived);
        let mut csr = ((offsets.len() - 1) as u32).to_le_bytes().to_vec();
        for v in offsets {
            csr.extend(v.to_le_bytes());
        }
        csr.extend(entries);
        let carries = crate::Tensor {
            name: "base.strassen_corrections".into(),
            offset: data.len() as u64,
            rows: rows / 2,
            cols: cols / 2,
            dtype: "strassen-csr-v1".into(),
            bytes: csr.len() as u64,
        };
        data.extend(csr);
        let model = "a".repeat(64);
        let hash = "b".repeat(64);
        let m = Manifest {
            version: 1,
            model: model.clone(),
            pack_hash: hash.clone(),
            bytes: data.len() as u64,
            tensors: vec![tensor, operand, carries],
        };
        let r = Request {
            version: 1,
            model,
            pack_hash: hash,
            input_hash: "c".repeat(64),
            step: 0,
            op: "linear_strassen_bf16".into(),
            tensor: "base".into(),
            dims: vec![n, rows, cols, 0],
            scalars: vec![],
            aux: vec![
                "base.strassen_i8".into(),
                "base.strassen_corrections".into(),
            ],
            encoding: "bf16-exact".into(),
        };
        (r, m, data, x, w, sw)
    }
    #[test]
    fn prepacked_integer_products_keep_block_scale_order_and_tail() {
        for n in [1, 7, 8, 9, 63, 64, 65] {
            let (r, m, data, x, w, sw) = fixture(n);
            let q = int8_kernel::quantize_rows(&x, n, 512).unwrap();
            let expected = int8_kernel::project(&q, &w, &sw, 8)
                .unwrap()
                .into_iter()
                .map(bf)
                .collect::<Vec<_>>();
            let got = evaluate(&r, &x, &m, &mut |offset, len| {
                Ok(data[offset as usize..offset as usize + len].to_vec())
            })
            .unwrap()
            .0;
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn invalid_shape_is_rejected_before_reading_any_weight() {
        let (mut r, m, _, x, _, _) = fixture(1);
        r.dims[3] = 8;
        let mut calls = 0;
        assert!(evaluate(&r, &x, &m, &mut |_, _| {
            calls += 1;
            Ok(vec![])
        })
        .is_err());
        assert_eq!(calls, 0);
    }
    #[test]
    fn corrupt_corrections_are_rejected_before_unsafe_dot() {
        let (r, m, mut data, x, _, _) = fixture(1);
        let t = &m.tensors[2];
        let records = 7 * 4 * 2;
        let header = 4 + 4 * (records + 1);
        data[t.offset as usize + header] = 128;
        assert!(evaluate(&r, &x, &m, &mut |offset, len| Ok(data
            [offset as usize..offset as usize + len]
            .to_vec()))
        .is_err());
    }
    #[cfg(feature = "experimental-strassen-prepared")]
    #[test]
    fn prepared_coefficients_match_original_and_reject_wrong_source_values() {
        for n in [1, 7, 8, 32, 64, 87] {
            let (mut r, m, data, x, w, sw) = fixture(n);
            let prepared = PreparedStrassen::prepare(&m, "base", |offset, len| Ok(data[offset as usize..offset as usize + len].to_vec())).unwrap();
            assert_eq!(prepared.bytes(), PreparedStrassen::required_bytes(&m, "base").unwrap());
            let expected: Vec<_> = int8_kernel::project(&int8_kernel::quantize_rows(&x, n, 512).unwrap(), &w, &sw, 8).unwrap().into_iter().map(bf).collect();
            let got = prepared.evaluate(&r, &x).unwrap();
            assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            r.pack_hash = "e".repeat(64);
            assert!(prepared.evaluate(&r, &x).is_err());
            let mut corrupt = data.clone(); corrupt[m.tensors[1].offset as usize] ^= 1;
            assert!(PreparedStrassen::prepare(&m, "base", |offset, len| Ok(corrupt[offset as usize..offset as usize + len].to_vec())).is_err());
        }
    }
}
