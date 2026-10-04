#[cfg(feature="experimental-projection-reuse")]
mod query_reply;
#[cfg(feature="experimental-projection-reuse")]
pub use query_reply::{EvaluatedReply,PayloadReply};
#[cfg(feature="experimental-attention-mlp-stream")]
mod attention_mlp_stream;
#[cfg(feature="experimental-mlp-delta-stream")]
mod delta_mlp_start;
#[cfg(feature="experimental-mlp-delta-stream")]
mod delta_head_continue;
#[cfg(feature="experimental-mlp-delta-stream")]
mod mlp_delta_stream;
#[cfg(feature="experimental-mlp-delta-stream")]
pub use mlp_delta_stream::PreparedMlpDeltaStream;
#[cfg(feature="experimental-int8-k-continue")]
mod int8_k_continue;
#[cfg(feature="experimental-int8-k-continue")]
pub use int8_k_continue::PreparedInt8K;
#[cfg(feature="experimental-mlp-stream")]
mod mlp_stream;
#[cfg(feature="experimental-mlp-stream")]
pub use mlp_stream::PreparedMlpStream;
#[cfg(feature="experimental-f32-k-continue")]
mod f32_k_continue;
#[cfg(feature="experimental-f32-k-continue")]
pub use f32_k_continue::PreparedF32K;
#[cfg(feature="experimental-f32-output-reuse")]
mod f32_output;
#[cfg(feature="experimental-mlp-delta-fusion")]
mod carry_planes;
#[cfg(feature="experimental-mlp-delta-fusion")]
mod mlp_delta_fusion;
#[cfg(feature="experimental-prefix-hybrid")]
pub mod prefix_hybrid_codec;
#[cfg(feature="experimental-prefix-hybrid")]
mod delta_hybrid;
#[cfg(feature="experimental-prepared-output-pairs")]
pub mod output_pairs;
#[cfg(feature="experimental-prepared-activation")]
pub mod prepared_activation;
#[cfg(feature="experimental-delta-full-log")]
mod delta_restore;
#[cfg(feature="experimental-delta-full-log")]
pub mod delta_log;
#[cfg(feature="experimental-delta-full-log")]
mod delta_recorded_simd;
#[cfg(feature="experimental-delta-full-log")]
mod delta_full_log;
#[cfg(feature="experimental-mlp-pipeline")]
mod mlp_pipeline;
#[cfg(feature="experimental-delta-finish")]
mod delta_finish;
pub mod prepared_weights;
use prepared_weights::{ByteOnly, LoadedWeight, WeightBuffer};
mod bf16_codec;
mod block_codec;
mod delta_simd;
mod delta_stage;
#[cfg(feature="experimental-delta-projected")]
mod delta_projected;
mod mlp_norm;
pub mod int8_kernel;
pub mod int8_token_kernel;
pub mod profile;
mod quantize_simd;
mod rope;
#[cfg(feature="experimental-prepared-rope")]
pub use rope::{prepare_fixed as prepare_rope_constants, fixed_bytes as rope_constant_bytes, clear_fixed as clear_rope_constants};
#[cfg(any(test, feature = "experimental-strassen-prepacked"))]
pub mod strassen_prepacked;
mod terminal;
#[cfg(feature = "experimental-attention-fusion")]
mod attention_fusion;
#[cfg(feature="experimental-attention-full")]
mod attention_full;
#[cfg(feature="experimental-terminal-attention")]
mod terminal_attention;
#[cfg(feature="experimental-terminal-tail")]
mod terminal_tail;
#[cfg(feature="experimental-terminal-tail")]
pub use terminal_tail::PreparedTail;
#[cfg(feature = "experimental-projection-reuse")]
mod projection_reuse;
#[cfg(feature = "experimental-projection-reuse")]
mod mlp_reuse;
#[cfg(feature = "experimental-projection-reuse")]
mod projection_codec;
// Qwen3.5 text primitives and dedicated decision head.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub type Result<T> = std::result::Result<T, String>;
pub const MAX_FLOATS: usize = 900_000;
pub fn lossless_encoding(encoding: &str) -> bool {
    matches!(encoding,"bf16-exact"|"bf16-block256-exact-v1") || {
        #[cfg(feature="experimental-mlp-pipeline")] {encoding==mlp_pipeline::NAME}
        #[cfg(not(feature="experimental-mlp-pipeline"))] {false}
    } || {
        #[cfg(feature = "experimental-projection-reuse")] {encoding == projection_codec::NAME}
        #[cfg(not(feature = "experimental-projection-reuse"))] {false}
    }
}
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
#[cfg(feature="experimental-prefix-start")]
mod prefix_start;
pub(crate) fn same_request(a:&Request,b:&Request)->bool {
    a.version==b.version && a.model==b.model && a.pack_hash==b.pack_hash
        && a.input_hash==b.input_hash && a.step==b.step && a.op==b.op
        && a.tensor==b.tensor && a.dims==b.dims && a.aux==b.aux && a.encoding==b.encoding
        && a.scalars.len()==b.scalars.len() && a.scalars.iter().zip(&b.scalars).all(|(a,b)|a.to_bits()==b.to_bits())
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
 #[cfg(feature="experimental-mlp-delta-fusion")]
 if mlp_delta_fusion::is_encoding(&r.encoding){return mlp_delta_fusion::reply_count(r).unwrap_or(0);}
 #[cfg(feature="experimental-attention-mlp-stream")]
 if r.encoding==attention_mlp_stream::NAME{return attention_mlp_stream::limit(r).unwrap_or(0);}
    #[cfg(feature="experimental-mlp-delta-stream")]
    if r.encoding==delta_mlp_start::NAME {return delta_mlp_start::limit(r).unwrap_or(0);}
    #[cfg(feature="experimental-mlp-delta-stream")]
    if r.encoding==mlp_delta_stream::NAME {return mlp_delta_stream::reply_count(r).unwrap_or(0);}
    #[cfg(feature="experimental-mlp-stream")]
    if r.encoding==mlp_stream::NAME {return mlp_stream::limit(r).unwrap_or(0);}
    #[cfg(feature="experimental-mlp-pipeline")]
    if r.encoding==mlp_pipeline::NAME {return mlp_pipeline::layout(r).map(|(_,c,q,t)|c+q+t).unwrap_or(0);}
    if r.op == "delta_heads_bf16" && r.encoding == "int8-block256-v1" {
        1_200_000
    } else {
        MAX_FLOATS
    }
}
/// Losslessly stores BF16-exact F32 values in two bytes and other F32 in four.
/// A bitmap marks full F32 exceptions; no rounding is performed by this codec.
fn frame_digest(version: u32, bytes: &[u8], encoding: bool) -> Result<[u8; 32]> {
    match version {
        1 | 3 => Ok(profile::measure(if encoding { "wire_encode_sha256" } else { "wire_decode_sha256" }, || Sha256::digest(bytes).into())),
        #[cfg(feature = "experimental-blake3")]
        2 => Ok(profile::measure(if encoding { "wire_encode_blake3" } else { "wire_decode_blake3" }, || *blake3::hash(bytes).as_bytes())),
        _ => Err("unsupported frame version".into()),
    }
}
fn checked_block_encoding(name:&str)->bool{match name{
    "bf16-block256-exact-v1"=>true,
    #[cfg(feature="experimental-prefix-hybrid")]
    delta_hybrid::NAME=>true,
    _=>false,
}}
pub fn encode(req: &Request, x: &[f32]) -> Result<Vec<u8>> {encode_impl(req,x,false)}
/// Use only after authenticated owner ingress. Version3 reply integrity is
/// provided by the query signature; the host seals it before durable storage.
#[cfg(feature="experimental-host-checksum")]
pub fn encode_signed_reply(req:&Request,x:&[f32])->Result<Vec<u8>>{encode_impl(req,x,true)}
fn encode_impl(req:&Request,x:&[f32],signed_transport:bool)->Result<Vec<u8>> {
    let limit = if lossless_encoding(&req.encoding) || req.encoding == "int8-block256-v1" {
        wire_float_limit(req)
    } else {
        450_000
    };
    #[cfg(feature="experimental-prefix-start")]
    let limit=if req.encoding==prefix_start::NAME {prefix_start::reply_count(req)?}else{limit};
    #[cfg(feature="experimental-mlp-delta-fusion")]
    let limit=if mlp_delta_fusion::is_encoding(&req.encoding) {mlp_delta_fusion::reply_count(req)?}else{limit};
    #[cfg(feature="experimental-mlp-stream")]
    let limit=if req.encoding==mlp_stream::NAME {mlp_stream::limit(req)?}else{limit};
    #[cfg(feature="experimental-mlp-delta-stream")]
    let limit=if req.encoding==delta_mlp_start::NAME {delta_mlp_start::limit(req)?}else{limit};
    #[cfg(feature="experimental-mlp-delta-stream")]
    let limit=if req.encoding==mlp_delta_stream::NAME {mlp_delta_stream::reply_count(req)?}else{limit};
    #[cfg(feature="experimental-attention-mlp-stream")]
    let limit=if req.encoding==attention_mlp_stream::NAME{attention_mlp_stream::limit(req)?}else{limit};
    if x.len() > limit || (!checked_block_encoding(&req.encoding) && !x.iter().all(|v| v.is_finite())) {
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
 #[cfg(feature="experimental-attention-mlp-stream")]
 attention_mlp_stream::NAME=>attention_mlp_stream::append(&mut b,req,x)?,
        #[cfg(feature="experimental-mlp-delta-stream")]
        delta_mlp_start::NAME=>delta_mlp_start::append(&mut b,req,x)?,
        #[cfg(feature="experimental-mlp-delta-stream")]
        mlp_delta_stream::NAME=>mlp_delta_stream::append_reply(&mut b,req,x)?,
        #[cfg(feature="experimental-mlp-stream")]
        mlp_stream::NAME=>mlp_stream::append(&mut b,req,x)?,
        #[cfg(feature="experimental-mlp-delta-fusion")]
        mlp_delta_fusion::NAME | mlp_delta_fusion::HUFFMAN_NAME => mlp_delta_fusion::append_reply(&mut b,req,x)?,
        #[cfg(feature="experimental-prefix-start")]
        prefix_start::NAME => prefix_start::append_reply(&mut b,req,x)?,
        "" => {
            for v in x {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        "bf16-block256-exact-v1" => block_codec::append(&mut b,x)?,
        #[cfg(feature="experimental-prefix-hybrid")]
        delta_hybrid::NAME => {b.push(0);block_codec::append(&mut b,x)?;},
        #[cfg(feature="experimental-mlp-pipeline")]
        mlp_pipeline::NAME=>mlp_pipeline::append(&mut b,req,x)?,
        #[cfg(feature = "experimental-projection-reuse")]
        projection_codec::NAME => projection_codec::append(&mut b,req,x)?,
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
    let digest = if signed_transport && req.version==3 {[0;32]}else{frame_digest(req.version, &b, true)?};
    b.extend_from_slice(&digest);
    Ok(b)
}
fn decode_envelope(b:&[u8])->Result<(Request,&[u8])>{decode_envelope_impl(b,false)}
fn decode_envelope_impl(b: &[u8], signed_transport:bool) -> Result<(Request, &[u8])> {
    if b.len() < 36 || b.len() > 2_000_000 {
        return Err("state size".into());
    }
    let n = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
    if n > 16_384 || n + 36 > b.len() {
        return Err("header/shape".into());
    }
    // Parse only the bounded header. Normal/file decoding verifies the whole
    // checksum before payload decoding. The explicit signed-transport context
    // may skip only version3 after owner ingress authentication.
    let r: Request = serde_json::from_slice(&b[4..4 + n]).map_err(|e| e.to_string())?;
    if !(signed_transport && r.version==3) && frame_digest(r.version, &b[..b.len() - 32], false)?.as_slice() != &b[b.len() - 32..] {
        return Err("checksum".into());
    }
    if r.model.len() != 64
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
    Ok((r,payload))
}
/// Only call after authenticating the owner of the complete IC message.
#[cfg(feature="experimental-host-checksum")]
pub fn decode_signed_input(b:&[u8])->Result<(Request,Vec<f32>)>{let(r,p)=decode_envelope_impl(b,true)?;let x=decode_values(&r,p)?;Ok((r,x))}
/// A checksum-verified local version3 request. Fields cannot be forged.
/// ```compile_fail
/// let mut bound: imajev_runtime::HostBoundRequest = todo!();
/// bound.request.version = 1;
/// ```
pub struct HostBoundRequest{request:Request}
pub fn is_host_bound_frame(b:&[u8])->bool {
    if b.len()<36 || b.len()>2_000_000{return false;}
    let n=u32::from_le_bytes(b[..4].try_into().unwrap())as usize;
    n<=16_384 && n+36<=b.len() && serde_json::from_slice::<Request>(&b[4..4+n]).is_ok_and(|r|r.version==3)
}
impl HostBoundRequest {
    pub fn verify_stored(b:&[u8])->Result<Self>{let(r,_)=decode_envelope(b)?;if r.version!=3{return Err("host-bound frame version".into());}Ok(Self{request:r})}
    /// Caller must first verify the complete IC query reply signature. This
    /// method binds progress/identity and adds the checksum used for storage.
    pub fn seal_verified_reply(&self,mut b:Vec<u8>)->Result<Vec<u8>> {
        let(r,_)=decode_envelope_impl(&b,true)?;let mut expected=self.request.clone();expected.step=expected.step.checked_add(1).ok_or("progress overflow")?;
        if !same_request(&r,&expected) || b[b.len()-32..]!=[0;32]{return Err("host-bound reply identity/footer".into());}
        let end=b.len()-32;let digest: [u8;32]=Sha256::digest(&b[..end]).into();b[end..].copy_from_slice(&digest);Ok(b)
    }
}
pub fn decode(b: &[u8]) -> Result<(Request, Vec<f32>)> {
    let (r,payload)=decode_envelope(b)?;
    let x=decode_values(&r,payload)?;
    Ok((r,x))
}
fn decode_values(r:&Request,payload:&[u8])->Result<Vec<f32>> {
    let x: Vec<f32> = match r.encoding.as_str() {
 #[cfg(feature="experimental-attention-mlp-stream")]
 attention_mlp_stream::NAME=>attention_mlp_stream::decode_reply(r,payload)?,
        #[cfg(feature="experimental-mlp-delta-stream")]
        delta_mlp_start::NAME=>delta_mlp_start::decode_reply(r,payload)?,
        #[cfg(feature="experimental-mlp-delta-stream")]
        mlp_delta_stream::NAME=>mlp_delta_stream::decode_reply(r,payload)?,
        #[cfg(feature="experimental-mlp-stream")]
        mlp_stream::NAME=>mlp_stream::decode_values(r,payload)?,
        #[cfg(feature="experimental-mlp-delta-fusion")]
        mlp_delta_fusion::NAME | mlp_delta_fusion::HUFFMAN_NAME => mlp_delta_fusion::decode_reply(r,payload)?,
        #[cfg(feature="experimental-prefix-start")]
        prefix_start::NAME => prefix_start::decode_reply(r,payload)?,
        "" => {
            if payload.len() % 4 != 0 || payload.len() / 4 > 450_000 {
                return Err("activation shape".into());
            }
            payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        "bf16-block256-exact-v1" => return block_codec::decode(payload),
        #[cfg(feature="experimental-prefix-hybrid")]
        delta_hybrid::NAME => {if payload.first()!=Some(&0){return Err("hybrid direction".into());}return block_codec::decode(&payload[1..]);},
        #[cfg(feature="experimental-mlp-pipeline")]
        mlp_pipeline::NAME=>mlp_pipeline::decode_values(r,payload)?,
        #[cfg(feature = "experimental-projection-reuse")]
        projection_codec::NAME => projection_codec::decode(&r,payload)?,
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
    Ok(x)
}
#[cfg(feature="experimental-projection-reuse")]
pub use projection_codec::PreparedProjection;
#[cfg(feature="experimental-projection-reuse")]
#[cfg(feature="experimental-mlp-pipeline")]
pub use mlp_pipeline::PreparedMlp;
#[cfg(feature="experimental-projection-reuse")]
pub enum DecodedQueryInput { #[cfg(feature="experimental-attention-mlp-stream")] AttentionMlp(attention_mlp_stream::PreparedAttentionMlp), #[cfg(feature="experimental-mlp-delta-stream")] DeltaMlpStart(delta_mlp_start::PreparedDeltaMlpStart), #[cfg(feature="experimental-mlp-delta-stream")] MlpDeltaStream(PreparedMlpDeltaStream), #[cfg(feature="experimental-int8-k-continue")] Int8K(PreparedInt8K), #[cfg(feature="experimental-mlp-stream")] MlpStream(PreparedMlpStream), #[cfg(feature="experimental-f32-k-continue")] F32K(PreparedF32K), #[cfg(feature="experimental-terminal-tail")] TerminalTail(PreparedTail), #[cfg(feature="experimental-mlp-delta-fusion")] MlpDelta(mlp_delta_fusion::PreparedCarry), #[cfg(feature="experimental-prefix-start")] PrefixStart(prefix_start::PreparedPrefixStart), #[cfg(feature="experimental-prefix-hybrid")] DeltaHybrid(delta_hybrid::PreparedDeltaHybrid), Values(Vec<f32>), Projection(PreparedProjection), #[cfg(feature="experimental-mlp-pipeline")] Mlp(PreparedMlp) }
#[cfg(feature="experimental-projection-reuse")]
impl DecodedQueryInput {
    pub fn values(&self)->Option<&[f32]> {match self { #[cfg(feature="experimental-attention-mlp-stream")] Self::AttentionMlp(_)=>None,#[cfg(feature="experimental-mlp-delta-stream")] Self::DeltaMlpStart(_)=>None, #[cfg(feature="experimental-mlp-delta-stream")] Self::MlpDeltaStream(_)=>None, #[cfg(feature="experimental-int8-k-continue")] Self::Int8K(_)=>None, #[cfg(feature="experimental-mlp-stream")] Self::MlpStream(_)=>None, #[cfg(feature="experimental-f32-k-continue")] Self::F32K(_)=>None, #[cfg(feature="experimental-terminal-tail")] Self::TerminalTail(_)=>None, #[cfg(feature="experimental-mlp-delta-fusion")] Self::MlpDelta(_)=>None, #[cfg(feature="experimental-prefix-start")] Self::PrefixStart(_)=>None, #[cfg(feature="experimental-prefix-hybrid")] Self::DeltaHybrid(_)=>None, Self::Values(v)=>Some(v),Self::Projection(_)=>None, #[cfg(feature="experimental-mlp-pipeline")] Self::Mlp(_)=>None}}
}
#[cfg(feature="experimental-projection-reuse")]
pub fn decode_query(b:&[u8])->Result<(Request,DecodedQueryInput)>{decode_query_impl(b,false)}
/// Requires authenticated owner ingress before invocation. Only version3 skips
/// the canister checksum pass; version1/2 still use their full checksum.
#[cfg(all(feature="experimental-projection-reuse",feature="experimental-host-checksum"))]
pub fn decode_signed_query(b:&[u8])->Result<(Request,DecodedQueryInput)>{decode_query_impl(b,true)}
#[cfg(feature="experimental-projection-reuse")]
fn decode_query_impl(b:&[u8],signed_transport:bool)->Result<(Request,DecodedQueryInput)> {
    let (r,payload)=decode_envelope_impl(b,signed_transport)?;
 #[cfg(feature="experimental-attention-mlp-stream")]
 if r.encoding==attention_mlp_stream::NAME{let input=attention_mlp_stream::PreparedAttentionMlp::decode(&r,payload)?;return Ok((r,DecodedQueryInput::AttentionMlp(input)));}
    #[cfg(feature="experimental-mlp-delta-stream")]
    if r.encoding==delta_mlp_start::NAME {let input=delta_mlp_start::PreparedDeltaMlpStart::decode(&r,payload)?;return Ok((r,DecodedQueryInput::DeltaMlpStart(input)));}
    #[cfg(feature="experimental-mlp-delta-stream")]
    if r.encoding==mlp_delta_stream::NAME {let input=PreparedMlpDeltaStream::decode(&r,payload)?;return Ok((r,DecodedQueryInput::MlpDeltaStream(input)));}
    #[cfg(feature="experimental-int8-k-continue")]
    if matches!(r.op.as_str(),"linear_integer_k_continue"|"linear_integer_k_finish") {let input=PreparedInt8K::decode(&r,payload)?;return Ok((r,DecodedQueryInput::Int8K(input)));}
    #[cfg(feature="experimental-mlp-stream")]
    if r.encoding==mlp_stream::NAME {let input=PreparedMlpStream::decode(&r,payload)?;return Ok((r,DecodedQueryInput::MlpStream(input)));}
    #[cfg(feature="experimental-f32-k-continue")]
    if r.op=="matmul_k_continue" {let input=PreparedF32K::decode(&r,payload)?;return Ok((r,DecodedQueryInput::F32K(input)));}
    #[cfg(feature="experimental-terminal-tail")]
    if r.op=="terminal_tail_integer" {let input=PreparedTail::decode(&r,payload)?;return Ok((r,DecodedQueryInput::TerminalTail(input)));}
    #[cfg(feature="experimental-mlp-delta-fusion")]
    if mlp_delta_fusion::is_encoding(&r.encoding) {let input=mlp_delta_fusion::PreparedCarry::decode(&r,payload)?;return Ok((r,DecodedQueryInput::MlpDelta(input)));}
    #[cfg(feature="experimental-prefix-start")]
    if r.encoding==prefix_start::NAME {
        let input=prefix_start::PreparedPrefixStart::decode(&r,payload)?;
        return Ok((r,DecodedQueryInput::PrefixStart(input)));
    }
    #[cfg(feature="experimental-prefix-hybrid")]
    if r.encoding==delta_hybrid::NAME {
        return Ok((r.clone(),DecodedQueryInput::DeltaHybrid(delta_hybrid::PreparedDeltaHybrid::decode(&r,payload)?)));
    }
    #[cfg(feature="experimental-mlp-pipeline")]
    if r.encoding==mlp_pipeline::NAME {
        let input=if r.op=="mlp_down_norm_prepared" || cfg!(feature="experimental-mlp-delta-fusion") && r.op=="mlp_down_norm_partial_prepared" {DecodedQueryInput::Mlp(PreparedMlp::decode(&r,payload)?)}
        else {if !(r.op=="mlp_prepare_down" || cfg!(feature="experimental-mlp-delta-fusion") && r.op=="mlp_prepare_partial_down")||payload.first()!=Some(&0){return Err("MLP pipeline request direction".into());}DecodedQueryInput::Values(decode_values(&r,payload)?)};
        return Ok((r,input));
    }
    let input=if r.encoding==projection_codec::NAME && matches!(r.op.as_str(),"lora_integer_reuse"|"mlp_gate_up_reuse") {
        DecodedQueryInput::Projection(PreparedProjection::decode(&r,payload)?)
    } else {DecodedQueryInput::Values(decode_values(&r,payload)?)};
    Ok((r,input))
}
#[cfg(feature="experimental-projection-reuse")]
pub fn evaluate_decoded_with_prepared_buffer<F,B>(r:&Request,input:&DecodedQueryInput,m:&Manifest,mut read:F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer,
{
    if r.model!=m.model || r.pack_hash!=m.pack_hash {return Err("model mismatch".into());}
    match input {
 #[cfg(feature="experimental-attention-mlp-stream")]
 DecodedQueryInput::AttentionMlp(_)=>Err("attention MLP requires owned evaluation".into()),
        #[cfg(feature="experimental-mlp-delta-stream")]
        DecodedQueryInput::DeltaMlpStart(_)=>Err("Delta MLP start requires owned evaluation".into()),
        #[cfg(feature="experimental-mlp-delta-stream")]
        DecodedQueryInput::MlpDeltaStream(_)=>Err("MLP Delta stream requires owned evaluation".into()),
        #[cfg(feature="experimental-f32-k-continue")]
        DecodedQueryInput::F32K(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-int8-k-continue")]
        DecodedQueryInput::Int8K(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-mlp-stream")]
        DecodedQueryInput::MlpStream(_)=>Err("MLP stream requires owned evaluation".into()),
        #[cfg(feature="experimental-terminal-tail")]
        DecodedQueryInput::TerminalTail(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-mlp-delta-fusion")]
        DecodedQueryInput::MlpDelta(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-prefix-start")]
        DecodedQueryInput::PrefixStart(input)=>input.evaluate(r,m,&mut read),
        DecodedQueryInput::Values(x)=>evaluate_with_prepared_buffer(r,x,m,read),
        #[cfg(feature="experimental-prefix-hybrid")]
        DecodedQueryInput::DeltaHybrid(input)=>input.evaluate(r,m,&mut read),
        DecodedQueryInput::Projection(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-mlp-pipeline")]
        DecodedQueryInput::Mlp(input)=>input.evaluate(r,m,&mut read),
    }
}
/// Consume a one-shot query operand, transferring prepared Delta state directly
/// into the recurrence. Borrowed callers retain the reusable API above.
#[cfg(feature="experimental-projection-reuse")]
pub fn evaluate_owned_decoded_with_prepared_buffer<F,B>(r:&Request,input:DecodedQueryInput,m:&Manifest,mut read:F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer,
{
    if r.model!=m.model || r.pack_hash!=m.pack_hash {return Err("model mismatch".into());}
    match input {
 #[cfg(feature="experimental-attention-mlp-stream")]
 DecodedQueryInput::AttentionMlp(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-mlp-delta-stream")]
        DecodedQueryInput::DeltaMlpStart(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-mlp-delta-stream")]
        DecodedQueryInput::MlpDeltaStream(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-mlp-stream")]
        DecodedQueryInput::MlpStream(input)=>input.evaluate(r,m,&mut read),
        #[cfg(feature="experimental-prefix-start")]
        DecodedQueryInput::PrefixStart(input)=>input.evaluate_owned(r,m,&mut read),
        #[cfg(feature="experimental-prefix-hybrid")]
        DecodedQueryInput::DeltaHybrid(input)=>input.evaluate_owned(r,m,&mut read),
        other=>evaluate_decoded_with_prepared_buffer(r,&other,m,read),
    }
}
/// Canister reply path avoids materializing integer carry as F32.
#[cfg(feature="experimental-projection-reuse")]
pub fn evaluate_owned_reply_with_prepared_buffer<F,B>(r:&Request,input:DecodedQueryInput,m:&Manifest,mut read:F)->Result<(EvaluatedReply,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    if r.model!=m.model || r.pack_hash!=m.pack_hash {return Err("model mismatch".into());}
    #[cfg(feature="experimental-direct-mlp-reply")]
    match input {
        DecodedQueryInput::MlpStream(input)=>return input.evaluate_reply(r,m,&mut read),
        #[cfg(feature="experimental-mlp-delta-stream")]
        DecodedQueryInput::MlpDeltaStream(input)=>return input.evaluate_reply(r,m,&mut read),
        #[cfg(feature="experimental-attention-mlp-stream")]
        DecodedQueryInput::AttentionMlp(input)=>return input.evaluate_reply(r,m,&mut read),
        other=>return evaluate_owned_decoded_with_prepared_buffer(r,other,m,read).map(|(v,b)|(EvaluatedReply::Values(v),b)),
    }
    #[cfg(not(feature="experimental-direct-mlp-reply"))]
    evaluate_owned_decoded_with_prepared_buffer(r,input,m,read).map(|(v,b)|(EvaluatedReply::Values(v),b))
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
    #[cfg(feature="experimental-prepared-activation")]
    if let Some(y)=prepared_activation::lookup::<3>(x){return y;}
    silu_original(x)
}
fn silu_original(x:f32)->f32 {
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
#[cfg(feature="experimental-matrix-tail")]
mod matrix_tail;
#[cfg(feature="experimental-lora-input-sharing")]
mod lora_input_sharing;
/// Diagnostic entry point for two distinct rank64 matrices sharing one F32 input.
#[cfg(feature="experimental-lora-input-sharing")]
pub fn shared_lora_a_pair(x:&[f32], first:&[f32], second:&[f32], n:usize, cols:usize)->Result<(Vec<f32>,Vec<f32>)> {
    lora_input_sharing::products(x,first,second,n,cols)
}
/// Diagnostic F32 kernel with grouped remainders and independent padded lanes.
#[cfg(feature="experimental-matrix-tail")]
pub fn matrix_tail(x:&[f32],w:&[f32],n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {
    matrix_tail::evaluate(x,w,n,rows,cols)
}
pub(crate) fn matrix_prepared(x:&[f32],w:&LoadedWeight,n:usize,rows:usize,cols:usize)->Option<Result<Vec<f32>>> {
    #[cfg(feature="experimental-f32-output-reuse")]
    if let LoadedWeight::Prepared(prepared)=w {return f32_output::project(x,prepared,n,rows,cols);}
    let _=(x,w,n,rows,cols);None
}
pub(crate) fn matrix_loaded(x:&[f32],w:&LoadedWeight,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {
    if let Some(result)=matrix_prepared(x,w,n,rows,cols) {return result;}
    matrix(x,w,n,rows,cols)
}
pub fn matrix(x: &[f32], w: &[f32], n: usize, rows: usize, cols: usize) -> Result<Vec<f32>> {
    #[cfg(feature="experimental-adopt-matrix-tail")]
    {
        let tile=if n>=128 && rows>=64 {64} else if n>=32 && rows>=64 {32} else {16};
        // With few output rows, packing dominates. A zero/four-token remainder
        // already uses the original vector path; the diagnostic regressed here.
        if rows<=64 && n%4==0 && n%tile<8 {
            return matrix_baseline(x,w,n,rows,cols);
        }
        return matrix_tail(x,w,n,rows,cols);
    }
    #[cfg(not(feature="experimental-adopt-matrix-tail"))]
    matrix_baseline(x,w,n,rows,cols)
}
/// Original kernel retained for same-Wasm diagnostic comparisons.
pub fn matrix_baseline(x: &[f32], w: &[f32], n: usize, rows: usize, cols: usize) -> Result<Vec<f32>> {
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
    #[cfg(feature="experimental-prepared-activation")]
    if let Some(y)=prepared_activation::lookup::<0>(x){return y;}
    bf_sigmoid_original(x)
}
fn bf_sigmoid_original(x:f32)->f32 {
    let y = bf(1.0 / bf(1.0 + bf(x.abs().exp())));
    if x < 0. {
        y
    } else {
        bf(1.0 - y)
    }
}
fn bf_silu(x: f32) -> f32 {
    #[cfg(feature="experimental-prepared-activation")]
    if let Some(y)=prepared_activation::lookup::<1>(x){return y;}
    bf_silu_original(x)
}
fn bf_silu_original(x:f32)->f32 {
    bf(x * bf_sigmoid_original(x))
}
fn bf_softplus(x: f32) -> f32 {
    #[cfg(feature="experimental-prepared-activation")]
    if let Some(y)=prepared_activation::lookup::<2>(x){return y;}
    bf_softplus_original(x)
}
fn bf_softplus_original(x:f32)->f32 {
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
    delta_impl::<true, false>(q, k, v, g, beta, state, dk, dv)
}

/// Compute Delta outputs with disposable state scratch. The final scratch contents
/// are unspecified; use `delta` when state must be carried to another query.
#[cfg(feature = "experimental-delta-no-writeback")]
pub fn delta_without_final_state(
    q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32],
    scratch: &mut [f32], dk: usize, dv: usize,
) -> Result<Vec<f32>> {
    delta_impl::<false, false>(q, k, v, g, beta, scratch, dk, dv)
}

/// Delta with key-major state scratch (`state[key * dv + value]`). Final state
/// remains key-major. Sum and rounding order match the value-major `delta` API.
#[cfg(feature = "experimental-delta-state-layout")]
pub fn delta_from_key_major(
    q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32],
    state: &mut [f32], dk: usize, dv: usize,
) -> Result<Vec<f32>> {
    delta_impl::<false, true>(q, k, v, g, beta, state, dk, dv)
}

fn delta_impl<const WRITEBACK: bool, const KEY_MAJOR: bool>(
    q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32],
    state: &mut [f32], dk: usize, dv: usize,
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
        return Ok(unsafe { delta_simd::run::<WRITEBACK, KEY_MAJOR>(q, k, v, g, beta, state, dk, dv) });
    }
    let mut y = vec![0.0; n * dv];
    for t in 0..n {
        for d in 0..dv {
            let mut mem = 0.0;
            for i in 0..dk {
                let at = if KEY_MAJOR { i * dv + d } else { d * dk + i };
                state[at] *= g[t];
                mem += state[at] * k[t * dk + i];
            }
            let update = (v[t * dv + d] - mem) * beta[t];
            let mut o = 0.0;
            for i in 0..dk {
                let at = if KEY_MAJOR { i * dv + d } else { d * dk + i };
                state[at] += k[t * dk + i] * update;
                o += state[at] * q[t * dk + i];
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
            let rate: Vec<_> = weight[..d[1]].iter().map(|v| -v.exp()).collect();
            for i in 0..count {
                y[i] = (rate[i % d[1]] * bf_softplus(bf(x[i] + weight[d[1] + i % d[1]])))
                    .exp();
                y[count + i] = bf_sigmoid(x[count + i]);
            }
            y
        }
        "rope" if d.len() == 4 => rope::transform(r, x)?,
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
        "delta_terminal_bf16" | "delta_heads_bf16" if d.len() == 4 => {
            let (n, dk, dv, h) = (d[0], d[1], d[2], d[3]);
            let terminal = r.op == "delta_terminal_bf16";
            if terminal && !crate::lossless_encoding(&r.encoding) {
                return Err("terminal delta requires lossless wire".into());
            }
            let mut contract = r.clone();
            contract.op = "delta_heads_bf16".into();
            int8_prefix(&contract, x.len())?;
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
                if !terminal {
                    states.extend(state);
                }
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
        "gqa_heads_bf16" | "gqa_suffix_bf16" if d.len() == 3 || d.len() == 4 => {
            let (n, width, heads) = (d[0], d[1], d[2]);
            let prefix = if r.op == "gqa_suffix_bf16" && d.len() == 4 {
                d[3]
            } else if r.op == "gqa_heads_bf16" && d.len() == 3 {
                0
            } else {
                return Err("GQA dimensions".into());
            };
            if !crate::lossless_encoding(&r.encoding)
                || n == 0
                || n > 512
                || prefix > 512
                || n + prefix > 512
                || width == 0
                || width > 256
                || heads == 0
                || heads > 16
                || (heads > 2 && heads % 4 != 0)
                || heads * (n * (n + 1) / 2 + n * prefix) * width > 75_000_000
                || x.len() != (heads * n + 2 * heads.div_ceil(4) * (n + prefix)) * width
            {
                return Err("GQA heads bounds".into());
            }
            let stride = n * width;
            let kv_stride = (n + prefix) * width;
            let (queries, kv) = x.split_at(heads * stride);
            let (keys, values) = kv.split_at(heads.div_ceil(4) * kv_stride);
            let mut single = r.clone();
            single.op = "attention_suffix_bf16".into();
            single.dims = vec![n, width, prefix];
            let mut input = Vec::with_capacity(3 * stride);
            let mut out = Vec::with_capacity(heads * stride);
            for head in 0..heads {
                input.clear();
                input.extend_from_slice(&queries[head * stride..(head + 1) * stride]);
                let group = head / 4;
                input.extend_from_slice(&keys[group * kv_stride..(group + 1) * kv_stride]);
                input.extend_from_slice(&values[group * kv_stride..(group + 1) * kv_stride]);
                out.extend(execute(&single, &input, &[])?);
            }
            out
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
        "rope_heads" if d.len() == 5 => rope::transform(r, x)?,
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
        "attention" | "attention_bf16" | "attention_suffix_bf16"
            if (d.len() == 2 || (r.op == "attention_suffix_bf16" && d.len() == 3))
                && d[0] > 0
                && d[0] <= 512
                && d[1] > 0
                && d[1] <= 256
                && d.get(2).copied().unwrap_or(0) <= 512
                && d[0] + d.get(2).copied().unwrap_or(0) <= 512
                && x.len() == (3 * d[0] + 2 * d.get(2).copied().unwrap_or(0)) * d[1] =>
        {
            let (n, h) = (d[0], d[1]);
            let prefix = d.get(2).copied().unwrap_or(0);
            let (q, z) = x.split_at(n * h);
            let (k, v) = z.split_at((n + prefix) * h);
            let mut out = vec![0.0; n * h];
            for t in 0..n {
                let mut scores = vec![0.0; prefix + t + 1];
                for s in 0..=prefix + t {
                    scores[s] = (0..h).map(|i| q[t * h + i] * k[s * h + i]).sum::<f32>()
                        / (h as f32).sqrt();
                    if r.op != "attention" {
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
                        if r.op != "attention" {
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
            if r.op != "attention" {
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
pub fn load_weight<F, B>(t: &Tensor, r: &Request, mut read: F) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<B>, B: AsRef<[u8]>,
{
    let (w, bytes) = load_prepared_weight(t, r, |offset, len| read(offset, len).map(ByteOnly))?;
    Ok((w.into_owned(), bytes))
}
fn load_prepared_weight<F, B>(t: &Tensor, r: &Request, mut read: F) -> Result<(LoadedWeight, u64)>
where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer,
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
    let raw = read(t.offset + (start * t.cols * bytes_per) as u64, n)?;
    if t.dtype == "f32" {
        if let Some(prepared) = raw.prepared_f32() {
            if prepared.len() != count { return Err("prepared weight shape".into()); }
            return Ok((LoadedWeight::Prepared(prepared), n as u64));
        }
    }
    let joined;
    let bytes = if t.dtype == "int8" {
        let scales = read(t.offset + (t.rows * t.cols + start * 4) as u64, rows * 4)?;
        n += scales.as_ref().len();
        joined = [raw.as_ref(), scales.as_ref()].concat();
        &joined[..]
    } else {
        raw.as_ref()
    };
    let tile = Tensor {
        name: t.name.clone(),
        offset: 0,
        rows,
        cols: t.cols,
        dtype: t.dtype.clone(),
        bytes: n as u64,
    };
    let w = unpack(&tile, bytes)?;
    if !w.iter().all(|v| v.is_finite()) {
        return Err("nonfinite weight".into());
    }
    Ok((LoadedWeight::Owned(w), n as u64))
}
#[cfg(test)]
mod prepared_reader_tests {
    use super::*;
    use prepared_weights::PreparedF32;
    struct PreparedBuffer(PreparedF32);
    impl AsRef<[u8]> for PreparedBuffer {
        fn as_ref(&self) -> &[u8] { panic!("prepared F32 must not decode bytes again") }
    }
    impl WeightBuffer for PreparedBuffer {
        fn prepared_f32(&self) -> Option<PreparedF32> { Some(self.0.clone()) }
    }
    #[test]
    fn prepared_nonzero_row_projection_matches_byte_reader_and_checks_size() {
        let w: Vec<_> = (0..8*256).map(|i| ((i * 37 % 101) as f32 - 50.) / 17.).collect();
        let bytes: Vec<_> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = Tensor { name: "adapter".into(), offset: 0, rows: 8, cols: 256, dtype: "f32".into(), bytes: bytes.len() as u64 };
        let m = Manifest { version: 1, model: "a".repeat(64), pack_hash: "b".repeat(64), bytes: t.bytes, tensors: vec![t] };
        let r: Request = serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"matmul","tensor":"adapter","dims":[7,4,256,3],"scalars":[]})).unwrap();
        let x: Vec<_> = (0..7*256).map(|i| (i as f32 - 20.) / 31.).collect();
        let prepared = PreparedF32::from_le_bytes(&bytes).unwrap();
        let a = evaluate_with_buffer(&r, &x, &m, |offset, len| Ok(bytes[offset as usize..offset as usize + len].to_vec())).unwrap();
        let b = evaluate_with_prepared_buffer(&r, &x, &m, |offset, len| Ok(PreparedBuffer(prepared.slice(offset as usize / 4, len / 4).unwrap()))).unwrap();
        assert_eq!(a.1, b.1);
        assert_eq!(a.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert!(evaluate_with_prepared_buffer(&r, &x, &m, |_, _| Ok(PreparedBuffer(prepared.slice(0, 1).unwrap()))).is_err());
    }
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

fn evaluate_integer<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F,
    prepared: Option<&int8_kernel::QuantizedRows>) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer,
{
    evaluate_integer_with_ax(r, x, m, read, prepared, None)
}
fn evaluate_integer_with_ax<F, B>(
    r: &Request,
    x: &[f32],
    m: &Manifest,
    mut read: &mut F,
    prepared: Option<&int8_kernel::QuantizedRows>,
    prepared_ax: Option<&[f32]>,
) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: WeightBuffer,
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
        || (n.checked_mul(cols) != Some(x.len()) && !(x.is_empty() && prepared_ax.is_some() && prepared.is_some_and(|q| q.rows() == n && q.cols() == cols)))
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
    let token_mode = cfg!(feature = "experimental-token-scale")
        && matches!(
            r.op.as_str(),
            "int8_matmul_token" | "linear_integer_token_bf16" | "lora_integer_token"
        );
    let with_lora = r.op == "lora_integer" || (token_mode && r.op == "lora_integer_token");
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
        if !r.aux.is_empty() || !r.scalars.is_empty() || prepared_ax.is_some() {
            return Err("integer matmul metadata".into());
        }
        None
    };
    let values = profile::measure("base_stable_read", || {
        read(tensor.offset + (start * cols) as u64, rows * cols)
    })?;
    #[cfg(feature="experimental-prepared-output-pairs")]
    let packed_weights = if !token_mode && n<=132 { values.prepared_output_pairs() } else { None };
    #[cfg(feature="experimental-prepared-output-pairs")]
    let cached_scales = packed_weights.as_ref().map(|w|w.scales());
    #[cfg(not(feature="experimental-prepared-output-pairs"))]
    let cached_scales: Option<&[f32]> = None;
    let scale_bytes;
    let fixed_scales;
    let owned_scales;
    let scales: &[f32] = if let Some(scales) = cached_scales {
        scales
    } else {
        scale_bytes = read(tensor.offset + (tensor.rows * cols + start * 4) as u64, rows * 4)?;
        fixed_scales = scale_bytes.prepared_f32();
        if let Some(ref fixed) = fixed_scales {
            if fixed.len() != rows { return Err("prepared integer scale shape".into()); }
            fixed
        } else {
            owned_scales = scale_bytes.as_ref().chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            &owned_scales
        }
    };
    // i8/u8 have identical size/alignment; every byte is a valid i8. The
    // immutable slice cannot outlive the reader buffer and is never mutated.
    let original_weights = || {
        let weight_bytes=values.as_ref();
        unsafe {std::slice::from_raw_parts(weight_bytes.as_ptr().cast::<i8>(),weight_bytes.len())}
    };
    let base = if token_mode {
        if prepared.is_some() {
            return Err("block quantization supplied to token projection".into());
        }
        let q = profile::measure("activation_quantize_token", || {
            int8_token_kernel::quantize(x, n, cols)
        })?;
        profile::measure("base_project_inclusive", || {
            int8_token_kernel::project(&q, original_weights(), &scales, rows)
        })?
    } else {
        let owned;
        let q = if let Some(q) = prepared {
            q
        } else {
            owned = profile::measure("activation_quantize", || {
                int8_kernel::quantize_rows(x, n, cols)
            })?;
            &owned
        };
        let result = profile::measure("base_project_inclusive", || {
            #[cfg(feature="experimental-prepared-output-pairs")]
            if q.rows()<=132 {
                if let Some(ref packed)=packed_weights {
                    if packed.cols()!=cols || packed.rows()!=rows {return Err("paired reader shape".into());}
                    return output_pairs::project_prepared(q,packed);
                }
            }
            #[cfg(not(all(target_arch="wasm32",feature="experimental-paired-only")))]
            {int8_kernel::project(&q, original_weights(), &scales, rows)}
            #[cfg(all(target_arch="wasm32",feature="experimental-paired-only"))]
            {Err("fixed paired weights required for this build".into())}
        })?;
        result
    };
    let mut read_bytes = (rows * cols + rows * 4) as u64;
    let out = if let Some((a, b)) = lora {
        let mut ar = r.clone();
        ar.op = "matmul".into();
        ar.dims = vec![n, a.rows, cols];
        let owned_ax;
        let (ax, ra) = if let Some(ax) = prepared_ax {
            if n.checked_mul(a.rows) != Some(ax.len()) || !ax.iter().all(|v| v.is_finite()) {
                return Err("prepared LoRA A shape/finite".into());
            }
            (ax, 0)
        } else {
            let (aw, ra) = profile::measure("lora_load_A", || load_prepared_weight(a, &ar, &mut read))?;
            owned_ax = profile::measure("lora_matmul_A", || matrix_loaded(x, &aw, n, a.rows, cols))?;
            (&owned_ax[..], ra)
        };
        ar.dims = vec![n, rows, a.rows, start];
        let (bw, rb) = profile::measure("lora_load_B", || load_prepared_weight(b, &ar, &mut read))?;
        read_bytes += ra + rb;
        let z = profile::measure("lora_matmul_B", || matrix_loaded(ax,&bw,n,rows,a.rows))?;
        base.into_iter()
            .zip(z)
            .map(|(v, z)| bf(bf(v) + bf(r.scalars[0] * z)))
            .collect::<Vec<_>>()
    } else if r.op == "linear_integer_bf16" || (token_mode && r.op == "linear_integer_token_bf16") {
        base.into_iter().map(bf).collect()
    } else {
        base
    };
    if !out.iter().all(|v| v.is_finite()) {
        return Err("integer fused output".into());
    }
    return Ok((out, read_bytes));
}
fn evaluate_mlp<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: WeightBuffer,
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
        request.op =
            if cfg!(feature = "experimental-token-scale") && r.op == "mlp_gate_up_integer_token" {
                "lora_integer_token"
            } else {
                "lora_integer"
            }
            .into();
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
    if work > 4_500_000_000 {
        return Err("fused MLP work limit".into());
    }
    let q = if cfg!(feature = "experimental-token-scale") && r.op == "mlp_gate_up_integer_token" {
        None
    } else {
        Some(profile::measure("activation_quantize", || {
            int8_kernel::quantize_rows(x, n, cols)
        })?)
    };
    #[cfg(feature="experimental-lora-input-sharing")]
    let shared = (4..=89).contains(&n) && q.is_some() && requests.iter().all(|r|
        m.tensors.iter().any(|t| t.name==r.aux[0] && t.rows==64));
    #[cfg(not(feature="experimental-lora-input-sharing"))]
    let shared = false;
    let (gate, up, rg, ru) = if shared {
        #[cfg(feature="experimental-lora-input-sharing")]
        {
            let (ga,ua,ra)=lora_input_sharing::pair_a(x,&requests,m,read)?;
            let (gate,rg)=evaluate_integer_with_ax(&requests[0],x,m,read,q.as_ref(),Some(&ga))?;
            let (up,ru)=evaluate_integer_with_ax(&requests[1],x,m,read,q.as_ref(),Some(&ua))?;
            (gate,up,rg+ra,ru)
        }
        #[cfg(not(feature="experimental-lora-input-sharing"))]
        unreachable!()
    } else {
        let (gate,rg)=evaluate_integer(&requests[0],x,m,read,q.as_ref())?;
        let (up,ru)=evaluate_integer(&requests[1],x,m,read,q.as_ref())?;
        (gate,up,rg,ru)
    };
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

/// Compatibility entry point for readers that return owned bytes.
pub fn evaluate_with_reader<F>(r: &Request, x: &[f32], m: &Manifest, read: F) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<Vec<u8>>,
{
    evaluate_with_buffer(r, x, m, read)
}

/// Identical pack dispatch for native validation and canister stable-memory tiles.
pub fn evaluate_with_buffer<F, B>(r: &Request, x: &[f32], m: &Manifest, mut read: F) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<B>, B: AsRef<[u8]>,
{
    evaluate_with_prepared_buffer(r, x, m, |offset, len| read(offset, len).map(ByteOnly))
}

pub fn evaluate_with_prepared_buffer<F, B>(r: &Request, x: &[f32], m: &Manifest, mut read: F) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer,
{
    if r.model != m.model || r.pack_hash != m.pack_hash {
        return Err("model mismatch".into());
    }
    #[cfg(feature="experimental-mlp-stream")]
    if r.encoding==mlp_stream::NAME {let b=encode(r,x)?;let(_,payload)=decode_envelope(&b)?;let input=PreparedMlpStream::decode(r,payload)?;return input.evaluate(r,m,&mut read);}
    #[cfg(feature="experimental-int8-k-continue")]
    if matches!(r.op.as_str(),"linear_integer_k_continue"|"linear_integer_k_finish") {return int8_k_continue::evaluate(r,x,m,&mut read);}
    #[cfg(feature="experimental-f32-k-continue")]
    if r.op=="matmul_k_continue" {return f32_k_continue::evaluate(r,x,m,&mut read);}
    #[cfg(feature = "experimental-projection-reuse")]
    if matches!(r.op.as_str(), "lora_integer_capture" | "lora_integer_reuse") {
        return projection_reuse::evaluate(r, x, m, &mut read);
    }
    #[cfg(feature = "experimental-projection-reuse")]
    if matches!(r.op.as_str(), "mlp_gate_up_capture" | "mlp_gate_up_reuse") {
        return mlp_reuse::evaluate(r, x, m, &mut read);
    }
    #[cfg(feature = "experimental-strassen-prepacked")]
    if r.op == "linear_strassen_bf16" {
        return strassen_prepacked::evaluate(r, x, m, &mut read);
    }
    #[cfg(feature="experimental-attention-full")]
    if r.op=="attention_full_integer" {return attention_full::evaluate(r,x,m,&mut read);}
    #[cfg(feature="experimental-terminal-tail")]
    if r.op=="terminal_tail_integer" {return terminal_tail::evaluate(r,x,m,&mut read);}
    #[cfg(feature="experimental-terminal-attention")]
    if r.op=="terminal_attention_mlp_integer" {return terminal_attention::evaluate(r,x,m,&mut read);}
    #[cfg(feature = "experimental-attention-fusion")]
    if matches!(r.op.as_str(),"attention_kv_integer"|"attention_q_gqa_integer") {
        return attention_fusion::evaluate(r,x,m,&mut read);
    }
    #[cfg(feature="experimental-mlp-pipeline")]
    if r.op=="mlp_prepare_down" {return mlp_pipeline::prepare(r,x,m,&mut read);}
    #[cfg(feature="experimental-mlp-delta-fusion")]
    if r.op=="mlp_prepare_partial_down" {return mlp_pipeline::prepare_partial(r,x,m,&mut read);}
    #[cfg(feature="experimental-mlp-full")]
    if r.op=="mlp_full_integer" {return mlp_pipeline::full(r,x,m,&mut read);}
    if r.op == "norm_rope_heads_bf16" {
        return rope::evaluate(r, x, m, &mut read);
    }
    if r.op == "terminal_mlp_integer" {
        return terminal::evaluate(r, x, m, &mut read);
    }
    #[cfg(feature="experimental-delta-full-log")]
    if r.op=="delta_full_log_integer" {return delta_full_log::evaluate(r,x,m,&mut read);}
    #[cfg(feature="experimental-delta-finish")]
    if r.op=="delta_project_finish" {return delta_finish::evaluate(r,x,m,&mut read);}
    #[cfg(feature="experimental-delta-projected")]
    if matches!(r.op.as_str(),"delta_project_capture"|"delta_project_reuse") {return delta_projected::evaluate(r,x,m,&mut read);}
    if r.op == "delta_input_integer" {
        return delta_stage::input(r, x, m, &mut read);
    }
    if r.op == "delta_gates_integer" {
        return delta_stage::gates(r, x, m, &mut read);
    }
    if r.op == "delta_stage_bf16" {
        return delta_stage::evaluate(r, x, m, &mut read);
    }
    if matches!(r.op.as_str(), "mlp_add_norm_integer" | "add_norm_chain_bf16") {
        return mlp_norm::evaluate(r, x, m, &mut read);
    }
    if r.op == "mlp_gate_up_integer"
        || (cfg!(feature = "experimental-token-scale") && r.op == "mlp_gate_up_integer_token")
    {
        return evaluate_mlp(r, x, m, &mut read);
    }
    if matches!(
        r.op.as_str(),
        "int8_matmul" | "linear_integer_bf16" | "lora_integer"
    ) || (cfg!(feature = "experimental-token-scale")
        && matches!(
            r.op.as_str(),
            "int8_matmul_token" | "linear_integer_token_bf16" | "lora_integer_token"
        ))
    {
        return evaluate_integer(r, x, m, &mut read, None);
    }
    let mut load = |name: &str, request: &Request| {
        let t = m
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or("unknown tensor")?;
        load_prepared_weight(t, request, &mut read)
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
            result.extend(row.iter().copied().map(bf));
            read += bytes;
        }
        return Ok((result, read));
    }
    if r.op == "delta_gates" {
        if r.aux.len() != 1 {
            return Err("gate tensors".into());
        }
        let (w, read_a) = load(&r.tensor, r)?;
        let (b, read_b) = load(&r.aux[0], r)?;
        let mut w = w.into_owned();
        w.extend_from_slice(&b);
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
            (LoadedWeight::Owned(vec![]), 0)
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

        let norm_name="model.language_model.layers.0.post_attention_layernorm.weight";
        tensors.push(Tensor{name:norm_name.into(),offset:pack.len() as u64,rows:1,cols:256,dtype:"f32".into(),bytes:1024});
        for _ in 0..256 {pack.extend(1f32.to_le_bytes());}
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
        let borrowed = |offset: u64, len: usize| {
            pack.get(offset as usize..offset as usize + len).ok_or("read bounds".into())
        };
        let (from_borrowed, borrowed_bytes) = evaluate_with_buffer(&r, &x, &manifest, borrowed).unwrap();
        let (from_owned, owned_bytes) = evaluate_with_reader(&r, &x, &manifest, read).unwrap();
        assert_eq!(from_borrowed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), from_owned.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(borrowed_bytes, owned_bytes);

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

        let mut fused=r.clone();fused.op="mlp_add_norm_integer".into();fused.encoding="bf16-exact".into();
        fused.aux.push(norm_name.into());fused.scalars.push(1e-6);
        let mut pair=x.clone();pair.extend(&x);
        let mut nr=r.clone();nr.op="add_norm_bf16".into();nr.dims=vec![2,256];nr.scalars=vec![1e-6];
        let norm=execute(&nr,&pair,&[1.;256]).unwrap();
        let expected=evaluate_with_reader(&r,&norm[512..],&manifest,read).unwrap().0;
        let actual=evaluate_with_reader(&fused,&pair,&manifest,read).unwrap().0;
        assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        fused.aux[1]="wrong.norm".into();
        assert!(evaluate_with_reader(&fused,&pair,&manifest,|_,_|panic!("must reject names before read")).is_err());
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

#[cfg(test)]
mod compact_head_tests {
    use super::*;
    fn request(op: &str, dims: Vec<usize>) -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"m","pack_hash":"p","input_hash":"i","step":0,"op":op,"tensor":"","dims":dims,"scalars":[],"encoding":"bf16-exact"})).unwrap()
    }
    #[test]
    fn suffix_gqa_preserves_causal_history_and_last_token() {
        let (n, w, h) = (9, 8, 8);
        let kv = h / 4;
        let q: Vec<_> = (0..h * n * w)
            .map(|i| bf((i as f32 * 0.037).sin()))
            .collect();
        let k: Vec<_> = (0..kv * n * w)
            .map(|i| bf((i as f32 * 0.053).cos()))
            .collect();
        let v: Vec<_> = (0..kv * n * w)
            .map(|i| bf((i as f32 * 0.013).sin()))
            .collect();
        let all = [q.clone(), k.clone(), v.clone()].concat();
        let baseline = execute(&request("gqa_heads_bf16", vec![n, w, h]), &all, &[]).unwrap();
        for prefix in [0, 3, 8] {
            let count = n - prefix;
            let qs: Vec<_> = (0..h)
                .flat_map(|head| {
                    q[(head * n + prefix) * w..(head + 1) * n * w]
                        .iter()
                        .copied()
                })
                .collect();
            let input = [qs, k.clone(), v.clone()].concat();
            let result = execute(
                &request("gqa_suffix_bf16", vec![count, w, h, prefix]),
                &input,
                &[],
            )
            .unwrap();
            let expected: Vec<_> = (0..h)
                .flat_map(|head| {
                    baseline[(head * n + prefix) * w..(head + 1) * n * w]
                        .iter()
                        .copied()
                })
                .collect();
            assert_eq!(
                result.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        for dims in [
            vec![1, w, h, 512],
            vec![1, w, h, usize::MAX],
            vec![1, w, h],
            vec![0, w, h, 1],
        ] {
            assert!(execute(&request("gqa_suffix_bf16", dims), &[], &[]).is_err());
        }
    }
    #[test]
    fn shared_gqa_matches_individual_heads_and_rejects_bad_layouts() {
        let n = 5;
        let width = 4;
        let heads = 8;
        let stride = n * width;
        let x: Vec<f32> = (0..(heads + heads / 2) * stride)
            .map(|i| (i % 11) as f32 / 16.)
            .collect();
        let r = request("gqa_heads_bf16", vec![n, width, heads]);
        let got = execute(&r, &x, &[]).unwrap();
        let mut expected = Vec::new();
        for head in 0..heads {
            let mut input = x[head * stride..(head + 1) * stride].to_vec();
            for offset in [heads, heads + heads / 4] {
                input.extend_from_slice(
                    &x[(offset + head / 4) * stride..(offset + head / 4 + 1) * stride],
                );
            }
            expected
                .extend(execute(&request("attention_bf16", vec![n, width]), &input, &[]).unwrap());
        }
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        for dims in [
            vec![0, 4, 8],
            vec![5, 4, 3],
            vec![5, 4, 20],
            vec![usize::MAX, 4, 8],
        ] {
            assert!(execute(&request("gqa_heads_bf16", dims), &x, &[]).is_err());
        }
        assert!(execute(&r, &x[..x.len() - 1], &[]).is_err());
        let mut lossy = r;
        lossy.encoding = "int8-block256-v1".into();
        assert!(execute(&lossy, &x, &[]).is_err());
    }
    #[test]
    fn terminal_delta_preserves_output_with_nonzero_initial_state() {
        let (n, k, v, h) = (3, 2, 2, 4);
        let mut x: Vec<f32> = (0..h * (n * (2 * k + v + 2) + k * v))
            .map(|i| ((i % 7) as f32 - 3.) / 32.)
            .collect();
        for head in 0..h {
            let tail = h * n * (2 * k + v) + head * (2 * n + k * v);
            x[tail..tail + n].fill(0.875);
            x[tail + n..tail + 2 * n].fill(0.5);
        }
        let full = execute(&request("delta_heads_bf16", vec![n, k, v, h]), &x, &[]).unwrap();
        let r = request("delta_terminal_bf16", vec![n, k, v, h]);
        let got = execute(&r, &x, &[]).unwrap();
        assert_eq!(got.len(), h * n * v);
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            full[..h * n * v]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(execute(&r, &x[..x.len() - 1], &[]).is_err());
        assert!(execute(
            &request("delta_terminal_bf16", vec![n, k, v, usize::MAX]),
            &x,
            &[]
        )
        .is_err());
        let mut lossy = r;
        lossy.encoding = "int8-block256-v1".into();
        assert!(execute(&lossy, &x, &[]).is_err());
    }
}

#[cfg(test)]
mod reader_buffer_contract_tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    // A safe AsRef implementation may return different valid slices on successive
    // calls. Bind one slice before converting its pointer and length together.
    struct ChangingBuffer {
        first: Vec<u8>,
        second: Vec<u8>,
        calls: Rc<Cell<usize>>,
    }
    impl AsRef<[u8]> for ChangingBuffer {
        fn as_ref(&self) -> &[u8] {
            let count = self.calls.get(); self.calls.set(count + 1);
            if count == 0 { &self.first } else { &self.second }
        }
    }
    #[test]
    fn signed_weight_view_binds_one_buffer_before_shape_validation() {
        let model = "a".repeat(64); let hash = "b".repeat(64);
        let m = Manifest { version: 1, model: model.clone(), pack_hash: hash.clone(), bytes: 2080,
            tensors: vec![Tensor { name: "weights".into(), offset: 0, rows: 8, cols: 256, dtype: "int8".into(), bytes: 2080 }] };
        let r: Request = serde_json::from_value(serde_json::json!({"version":1,"model":model,"pack_hash":hash,"input_hash":"c".repeat(64),"step":0,"op":"linear_integer_bf16","tensor":"weights","dims":[1,8,256,0],"scalars":[],"aux":[]})).unwrap();
        let calls = Rc::new(Cell::new(0));
        let result = evaluate_with_buffer(&r, &[1.; 256], &m, |offset, len| {
            if offset == 0 {
                Ok(ChangingBuffer { first: vec![1], second: vec![1; len], calls: calls.clone() })
            } else {
                let bytes: Vec<_> = (0..8).flat_map(|_| 0.01f32.to_le_bytes()).collect();
                Ok(ChangingBuffer { first: bytes.clone(), second: bytes, calls: Rc::new(Cell::new(0)) })
            }
        });
        assert!(result.is_err());
        assert_eq!(calls.get(), 1);
    }
}

#[cfg(test)]
mod frame_checksum_tests {
    use super::*;
    fn request(version: u32, encoding: &str) -> Request {
        serde_json::from_value(serde_json::json!({"version":version,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"bf16","tensor":"","dims":[],"scalars":[],"encoding":encoding})).unwrap()
    }
    #[test]
    fn unsupported_versions_cannot_encode_or_decode() {
        assert!(encode(&request(4,""), &[1.]).is_err());
        let mut b = encode(&request(1,""), &[1.]).unwrap();
        let at = b.windows(11).position(|w| w == b"\"version\":1").unwrap()+10;
        b[at]=b'4';
        assert!(decode(&b).is_err());
        #[cfg(not(feature="experimental-blake3"))]
        assert!(encode(&request(2,""), &[1.]).is_err());
    }
    #[cfg(feature="experimental-blake3")]
    #[test]
    fn both_checksums_preserve_bits_and_reject_corruption_and_downgrades() {
        let x=[0.,-0.,1.0000001,f32::from_bits(1),4.25];
        for codec in ["","bf16-exact","bf16-block256-exact-v1"] {
            let old=encode(&request(1,codec), &x).unwrap();
            let new=encode(&request(2,codec), &x).unwrap();
            assert_eq!(old.len(),new.len());
            for b in [&old,&new] {
                let (_, got)=decode(b).unwrap();
                assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),x.map(f32::to_bits));
                for offset in [4,b.len()-33,b.len()-1] {
                    let mut corrupt=b.clone();corrupt[offset]^=1;assert!(decode(&corrupt).is_err());
                }
            }
            let mut downgraded=new.clone();
            let at=downgraded.windows(11).position(|w| w==b"\"version\":2").unwrap()+10;
            downgraded[at]=b'1';assert!(decode(&downgraded).is_err());
            let end=new.len()-32;
            let mut wrong=new[..end].to_vec();wrong.extend_from_slice(&Sha256::digest(&new[..end]));
            assert!(decode(&wrong).is_err());
        }
    }
}

#[cfg(all(test,feature="experimental-delta-state-layout"))]
mod delta_state_layout_tests;

#[cfg(test)]mod host_checksum_tests {
 use super::*;
 fn request()->Request{serde_json::from_value(serde_json::json!({"version":3,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":7,"op":"bf16","tensor":"","dims":[],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap()}
 #[test]fn stored_v3_is_checked_and_reply_progress_is_bound(){
  let r=request();let input=encode(&r,&[1.,-0.,f32::from_bits(1)]).unwrap();assert!(is_host_bound_frame(&input));let bound=HostBoundRequest::verify_stored(&input).unwrap();
  for at in [4,input.len()-33,input.len()-1]{let mut b=input.clone();b[at]^=1;assert!(HostBoundRequest::verify_stored(&b).is_err());}
  let mut reply=r.clone();reply.step+=1;let mut wire=encode(&reply,&[1.,-0.,f32::from_bits(1)]).unwrap();let end=wire.len()-32;wire[end..].fill(0);assert!(decode(&wire).is_err());let sealed=bound.seal_verified_reply(wire.clone()).unwrap();let(h,x)=decode(&sealed).unwrap();assert_eq!(h.step,8);assert_eq!(x[1].to_bits(),(-0f32).to_bits());assert!(bound.seal_verified_reply(sealed).is_err());
  for field in 0..11{let mut other=reply.clone();match field{0=>other.version=1,1=>other.model="d".repeat(64),2=>other.pack_hash="d".repeat(64),3=>other.input_hash="d".repeat(64),4=>other.step+=1,5=>other.op="embed".into(),6=>other.tensor="other".into(),7=>other.dims.push(1),8=>other.scalars.push(1.),9=>other.aux.push("other".into()),_=>other.encoding="bf16-exact".into()};let mut b=encode(&other,&[1.]).unwrap();let end=b.len()-32;b[end..].fill(0);assert!(bound.seal_verified_reply(b).is_err());}
 }
 #[cfg(feature="experimental-host-checksum")]
 #[test]fn trusted_context_only_skips_v3_checksum_not_input_validation(){
  for version in [1,2,3]{if version==2&&!cfg!(feature="experimental-blake3"){continue;}let mut r=request();r.version=version;let mut b=encode(&r,&[1.]).unwrap();let end=b.len()-32;b[end..].fill(0);assert!(decode(&b).is_err());assert_eq!(decode_signed_input(&b).is_ok(),version==3);}
  let r=request();let mut b=encode(&r,&[1.]).unwrap();let n=u32::from_le_bytes(b[..4].try_into().unwrap())as usize;let pos=4+n+5;b[pos..pos+2].copy_from_slice(&0x7fc0u16.to_le_bytes());assert!(decode_signed_input(&b).is_err());
 }
}
