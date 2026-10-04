use candid::{CandidType, Principal};
use imajev_runtime::Manifest;
#[cfg(feature="experimental-host-checksum")]
use imajev_runtime::{decode_signed_input as decode,encode_signed_reply as encode};
#[cfg(not(feature="experimental-host-checksum"))]
use imajev_runtime::{decode,encode};
#[cfg(all(feature="experimental-projection-reuse",feature="experimental-host-checksum"))]
use imajev_runtime::decode_signed_query as decode_query;
#[cfg(all(feature="experimental-projection-reuse",not(feature="experimental-host-checksum")))]
use imajev_runtime::decode_query;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
mod weight_cache;
#[derive(Default)]
struct Store {
    owner: Option<Principal>,
    manifest: Option<Manifest>,
    received: u64,
    ready: bool,
    digest: Sha256,
    chunks: std::collections::BTreeMap<u64, Vec<u8>>,
    hashed: u64,
    weight_cache: weight_cache::WeightCache,
}
thread_local! {static STORE:RefCell<Store>=RefCell::new(Store::default());}
fn owner() {
    STORE.with(|s| {
        assert_eq!(
            s.borrow().owner,
            Some(ic_cdk::api::msg_caller()),
            "owner only"
        )
    });
}
#[ic_cdk::init]
fn init(owner: Principal) {
    assert_ne!(owner, Principal::anonymous());
    STORE.with(|s| s.borrow_mut().owner = Some(owner));
}
#[cfg(feature = "experimental-byte-buffer")]
type StateBytes = serde_bytes::ByteBuf;
#[cfg(not(feature = "experimental-byte-buffer"))]
type StateBytes = Vec<u8>;

#[derive(CandidType, Deserialize)]
struct Measurement {
    state: StateBytes,
    instructions: u64,
    stable_read_bytes: u64,
    heap_pages: u64,
    stable_pages: u64,
}
#[ic_cdk::update]
fn prepare(manifest: String) -> std::result::Result<(), String> {
    owner();
    let m: Manifest = serde_json::from_str(&manifest).map_err(|e| e.to_string())?;
    if m.version != 1
        || m.model.len() != 64
        || m.pack_hash.len() != 64
        || m.bytes > 24 * 1024 * 1024 * 1024
        || m.tensors.len() > 2000
    {
        return Err("manifest bounds".into());
    }
    let mut end = 0;
    let mut names = std::collections::BTreeSet::new();
    for t in &m.tensors {
        if !names.insert(&t.name)
            || t.rows == 0
            || t.cols == 0
            || t.rows.checked_mul(t.cols).is_none()
            || t.offset < end
            || t.bytes == 0
            || t.offset.checked_add(t.bytes).is_none()
            || t.offset + t.bytes > m.bytes
        {
            return Err("tensor bounds".into());
        }
        end = t.offset + t.bytes;
    }
    if STORE.with(|s| s.borrow().ready) {
        return Err("already prepared".into());
    }
    let pages = (m.bytes + 65535) / 65536;
    let old = ic_cdk::api::stable_size();
    if pages > old && ic_cdk::api::stable_grow(pages - old) == u64::MAX {
        return Err("stable grow".into());
    }
    STORE.with(|s| {
        let mut s = s.borrow_mut();
        if s.ready {
            return Err("already prepared".into());
        }
        s.manifest = Some(m);
        s.received = 0;
        s.digest = Sha256::new();
        s.chunks.clear();
        s.hashed = 0;
        Ok(())
    })
}
#[ic_cdk::update]
fn upload(offset: u64, bytes: Vec<u8>, sha: Vec<u8>) -> std::result::Result<u64, String> {
    owner();
    if bytes.is_empty() || bytes.len() > 1_800_000 || Sha256::digest(&bytes).as_slice() != sha {
        return Err("chunk checksum/size".into());
    }
    STORE.with(|s| {
        let mut s = s.borrow_mut();
        let m = s.manifest.as_ref().ok_or("not prepared")?;
        if s.ready
            || !s.chunks.is_empty()
            || offset.checked_add(bytes.len() as u64).is_none()
            || offset + bytes.len() as u64 > m.bytes
        {
            return Err("upload range".into());
        }
        if offset < s.received {
            let mut existing = vec![0; bytes.len()];
            ic_cdk::api::stable_read(offset, &mut existing);
            if existing == bytes {
                return Ok(s.received);
            }
            return Err("retry mismatch".into());
        }
        if offset != s.received {
            return Err("upload order".into());
        }
        ic_cdk::api::stable_write(offset, &bytes);
        s.digest.update(&bytes);
        s.received += bytes.len() as u64;
        s.hashed = s.received;
        Ok(s.received)
    })
}
const CHUNK: u64 = 1_800_000;
#[derive(CandidType, Deserialize)]
struct PackStatus {
    model: String,
    pack_hash: String,
    bytes: u64,
    received: u64,
    hashed: u64,
    ready: bool,
    chunks: Vec<u64>,
}
#[ic_cdk::query]
fn pack_status() -> PackStatus {
    owner();
    STORE.with(|s| {
        let s = s.borrow();
        PackStatus {
            model: s
                .manifest
                .as_ref()
                .map(|m| m.model.clone())
                .unwrap_or_default(),
            pack_hash: s
                .manifest
                .as_ref()
                .map(|m| m.pack_hash.clone())
                .unwrap_or_default(),
            bytes: s.manifest.as_ref().map(|m| m.bytes).unwrap_or(0),
            received: s.received,
            hashed: s.hashed,
            ready: s.ready,
            chunks: s.chunks.keys().copied().collect(),
        }
    })
}
#[ic_cdk::update]
fn upload_chunk(offset: u64, bytes: Vec<u8>, sha: Vec<u8>) -> std::result::Result<u64, String> {
    owner();
    if bytes.is_empty() || bytes.len() as u64 > CHUNK || Sha256::digest(&bytes).as_slice() != sha {
        return Err("chunk checksum/size".into());
    }
    STORE.with(|s| {
        let mut s = s.borrow_mut();
        let end = s.manifest.as_ref().ok_or("not prepared")?.bytes;
        if s.ready
            || offset % CHUNK != 0
            || offset >= end
            || bytes.len() as u64 != (end - offset).min(CHUNK)
        {
            return Err("chunk range/alignment".into());
        }
        if let Some(previous) = s.chunks.get(&offset) {
            return if previous == &sha {
                Ok(s.received)
            } else {
                Err("retry mismatch".into())
            };
        }
        if s.received > 0 && s.chunks.is_empty() {
            return Err("upload mode mismatch".into());
        }
        ic_cdk::api::stable_write(offset, &bytes);
        s.chunks.insert(offset, sha);
        while s.received < end && s.chunks.contains_key(&s.received) {
            s.received += (end - s.received).min(CHUNK);
        }
        Ok(s.received)
    })
}
#[ic_cdk::update]
fn hash_pack(max_bytes: u64) -> std::result::Result<(u64, bool), String> {
    owner();
    if max_bytes == 0 || max_bytes > 8_000_000 {
        return Err("hash batch bounds".into());
    }
    STORE.with(|s| {
        let mut s = s.borrow_mut();
        let m = s.manifest.as_ref().ok_or("not prepared")?;
        let end = m.bytes;
        let expected = m.pack_hash.clone();
        if s.ready {
            return Ok((s.hashed, true));
        }
        let n = (s.received - s.hashed).min(max_bytes) as usize;
        if n == 0 {
            return Err("no contiguous bytes to hash".into());
        }
        let mut bytes = vec![0; n];
        ic_cdk::api::stable_read(s.hashed, &mut bytes);
        s.digest.update(&bytes);
        s.hashed += n as u64;
        if s.hashed == end {
            let actual = s
                .digest
                .clone()
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            if actual != expected {
                return Err("pack hash mismatch".into());
            }
            s.ready = true;
        }
        Ok((s.hashed, s.ready))
    })
}
#[ic_cdk::update]
fn seal() -> std::result::Result<(), String> {
    owner();
    STORE.with(|s| {
        let mut s = s.borrow_mut();
        if s.received != s.manifest.as_ref().ok_or("not prepared")?.bytes {
            return Err("incomplete".into());
        }
        let actual = s
            .digest
            .clone()
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if actual != s.manifest.as_ref().unwrap().pack_hash {
            return Err("pack hash mismatch".into());
        }
        s.ready = true;
        Ok(())
    })
}
#[ic_cdk::query]
fn step(state: StateBytes) -> std::result::Result<Measurement, String> {
    owner();
    let start = ic_cdk::api::performance_counter(0u32);
    #[cfg(feature = "experimental-projection-reuse")]
    let (mut r, input) = decode_query(&state)?;
    #[cfg(feature = "experimental-projection-reuse")]
    let (y, read) = evaluate_decoded(&r, input)?;
    #[cfg(not(feature = "experimental-projection-reuse"))]
    let (mut r, x) = decode(&state)?;
    #[cfg(not(feature = "experimental-projection-reuse"))]
    let (y, read) = evaluate(&r, &x)?;
    r.step = r.step.checked_add(1).ok_or("progress overflow")?;
    #[cfg(feature="experimental-projection-reuse")]
    let state = y.encode(&r,cfg!(feature="experimental-host-checksum"))?;
    #[cfg(not(feature="experimental-projection-reuse"))]
    let state = encode(&r, &y)?;
    #[cfg(target_arch = "wasm32")]
    let heap_pages = core::arch::wasm32::memory_size(0) as u64;
    #[cfg(not(target_arch = "wasm32"))]
    let heap_pages = 0;
    Ok(Measurement {
        state: state.into(),
        instructions: ic_cdk::api::performance_counter(0u32) - start,
        stable_read_bytes: read,
        heap_pages,
        stable_pages: ic_cdk::api::stable_size(),
    })
}
#[derive(CandidType, Deserialize)]
struct ProfileMeasurement {
    measurement: Measurement,
    spans: Vec<(String, u64, u64)>,
}
#[ic_cdk::query]
fn profile_step(state: StateBytes) -> std::result::Result<ProfileMeasurement, String> {
    owner();
    if !cfg!(feature = "instruction-profile") {
        return Err("instruction-profile feature is disabled".into());
    }
    imajev_runtime::profile::start(|| ic_cdk::api::performance_counter(0));
    let result = (|| {
        let start = ic_cdk::api::performance_counter(0);
        #[cfg(feature = "experimental-projection-reuse")]
        let (mut r, input) = imajev_runtime::profile::measure("wire_decode", || decode_query(&state))?;
        #[cfg(feature = "experimental-projection-reuse")]
        let (y, read) = imajev_runtime::profile::measure("evaluate_inclusive", || evaluate_decoded(&r, input))?;
        #[cfg(not(feature = "experimental-projection-reuse"))]
        let (mut r, x) = imajev_runtime::profile::measure("wire_decode", || decode(&state))?;
        #[cfg(not(feature = "experimental-projection-reuse"))]
        let (y, read) =
            imajev_runtime::profile::measure("evaluate_inclusive", || evaluate(&r, &x))?;
        r.step = r.step.checked_add(1).ok_or("progress overflow")?;
        #[cfg(feature="experimental-projection-reuse")]
        let state = imajev_runtime::profile::measure("wire_encode", || y.encode(&r,cfg!(feature="experimental-host-checksum")))?;
        #[cfg(not(feature="experimental-projection-reuse"))]
        let state = imajev_runtime::profile::measure("wire_encode", || encode(&r, &y))?;
        #[cfg(target_arch = "wasm32")]
        let heap_pages = core::arch::wasm32::memory_size(0) as u64;
        #[cfg(not(target_arch = "wasm32"))]
        let heap_pages = 0;
        Ok(Measurement {
            state: state.into(),
            instructions: ic_cdk::api::performance_counter(0) - start,
            stable_read_bytes: read,
            heap_pages,
            stable_pages: ic_cdk::api::stable_size(),
        })
    })();
    let spans = imajev_runtime::profile::finish();
    result.map(|measurement| ProfileMeasurement { measurement, spans })
}
#[cfg(feature = "experimental-projection-reuse")]
fn evaluate_decoded(r:&imajev_runtime::Request,input:imajev_runtime::DecodedQueryInput)->std::result::Result<(imajev_runtime::EvaluatedReply,u64),String> {
    if let imajev_runtime::DecodedQueryInput::Values(x)=input {return evaluate(r,&x).map(|(v,b)|(imajev_runtime::EvaluatedReply::Values(v),b));}
    STORE.with(|s| {
        let s=s.borrow();
        if !s.ready {return Err("not ready".into());}
        let m=s.manifest.as_ref().ok_or("missing manifest")?;
        let mut physical_read_bytes=0u64;
        let (output,_)=imajev_runtime::evaluate_owned_reply_with_prepared_buffer(r,input,m,|offset,len| {
            if offset.checked_add(len as u64).is_none_or(|end| end>m.bytes) {return Err("weight read range".into());}
            if let Some(bytes)=s.weight_cache.read(offset,len) {return Ok(bytes);}
            let mut bytes=vec![0;len];
            ic_cdk::api::stable_read(offset,&mut bytes);
            physical_read_bytes=physical_read_bytes.checked_add(len as u64).ok_or("read byte counter overflow")?;
            Ok(weight_cache::ReadBuffer::Owned(bytes))
        })?;
        Ok((output,physical_read_bytes))
    })
}
fn evaluate(
    r: &imajev_runtime::Request,
    x: &[f32],
) -> std::result::Result<(Vec<f32>, u64), String> {
    STORE.with(|s| {
        let s = s.borrow();
        if !s.ready {
            return Err("not ready".into());
        }
        let m = s.manifest.as_ref().ok_or("missing manifest")?;
        if r.model != m.model || r.pack_hash != m.pack_hash {
            return Err("model mismatch".into());
        }
        #[cfg(feature = "experimental-strassen-prepared")]
        if r.op == "linear_strassen_bf16" {
            if let Some(weights) = s.weight_cache.strassen(&r.tensor) {
                return Ok((weights.evaluate(r, x)?, 0));
            }
        }
        let mut physical_read_bytes = 0u64;
        let (output, _) = imajev_runtime::evaluate_with_prepared_buffer(r, x, m, |offset, len| {
            if offset.checked_add(len as u64).is_none_or(|end| end > m.bytes) {
                return Err("weight read range".into());
            }
            if let Some(bytes) = s.weight_cache.read(offset, len) {
                return Ok(bytes);
            }
            let mut bytes = vec![0; len];
            ic_cdk::api::stable_read(offset, &mut bytes);
            physical_read_bytes = physical_read_bytes.checked_add(len as u64).ok_or("read byte counter overflow")?;
            Ok(weight_cache::ReadBuffer::Owned(bytes))
        })?;
        Ok((output, physical_read_bytes))
    })
}
#[derive(CandidType, Deserialize)]
struct TerminalDecisionMeasurement {
    measurement: Measurement,
    decision: ChoiceResult,
}
fn validate_terminal_decision(r: &imajev_runtime::Request, options: &[String]) -> Result<(), String> {
    if r.op == "mlp_stream_complete_terminal" {
        let step = if cfg!(feature="experimental-mlp-half") {128} else {256};
        if !cfg!(feature="experimental-terminal-stream")
            || r.tensor != "model.language_model.layers.30.post_attention_layernorm.weight"
            || r.encoding != "mlp-attention-finish-exact-v1"
            || r.dims.len() != 3 || !(1..=89).contains(&r.dims[0])
            || r.dims[1] > 132 || r.dims[2] == 0 || r.dims[2] >= 9216 || r.dims[2] % step != 0
            || r.aux != ["model.language_model.layers.31.input_layernorm.weight"]
            || r.scalars.len() != 2 || r.scalars[0].to_bits() != 2f32.to_bits()
            || r.scalars[1].to_bits() != 1e-6f32.to_bits() || r.step == u64::MAX {
            return Err("terminal stream decision metadata".into());
        }
        imajev_runtime::decide_candidates(options,&vec![0.;options.len().min(7)+1],1.3051569717552742)?;
        return Ok(());
    }
    let tail=r.op=="terminal_tail_integer" && cfg!(feature="experimental-terminal-tail");
    if !(r.op=="terminal_attention_mlp_integer" || tail)
        || r.tensor != if tail {"model.language_model.layers.30.post_attention_layernorm.weight"}else{"model.language_model.layers.31.self_attn.q_proj.weight"}
        || !(r.dims.len()==2 || (tail && r.dims.len()==3 && r.dims[2]==0)) || !(1..=if tail{89}else{132}).contains(&r.dims[0])
        || r.dims[1] > 512 || r.dims[0] + r.dims[1] > 512
        || !r.aux.is_empty() || !r.scalars.is_empty()
        || !matches!(r.encoding.as_str(), "bf16-exact" | "bf16-block256-exact-v1")
        || r.step == u64::MAX {
        return Err("terminal decision metadata".into());
    }
    // The same checked option contract used by the dedicated decision query.
    imajev_runtime::decide_candidates(options, &vec![0.; options.len().min(7)+1], 1.3051569717552742)?;
    Ok(())
}
#[ic_cdk::query]
fn terminal_step_decision(state: StateBytes, options: Vec<String>) -> Result<TerminalDecisionMeasurement, String> {
    owner();
    let start = ic_cdk::api::performance_counter(0);
    if !cfg!(feature="experimental-terminal-attention") {return Err("terminal attention feature is disabled".into());}
    #[cfg(feature="experimental-projection-reuse")]
    let(mut r,input)=decode_query(&state)?;
    #[cfg(not(feature="experimental-projection-reuse"))]
    let(mut r,x)=decode(&state)?;
    validate_terminal_decision(&r, &options)?;
    #[cfg(feature="experimental-projection-reuse")]
    let(y,mut read)=evaluate_decoded(&r,input)?;
    #[cfg(feature="experimental-projection-reuse")]
    let y=y.into_values()?;
    #[cfg(not(feature="experimental-projection-reuse"))]
    let(y,mut read)=evaluate(&r,&x)?;
    let offset=if r.op=="terminal_tail_integer" && r.dims.len()==2 {r.dims[0]*2560}else{0};
    if y.len() != offset + 5120 + r.dims[0]*2048 {return Err("terminal decision output shape".into());}
    let decision_start = ic_cdk::api::performance_counter(0);
    let mut dr = r.clone();
    dr.op="matmul".into();dr.tensor="readout-f32".into();
    dr.dims=vec![1, options.len()+1, 2560, 0];dr.aux.clear();dr.scalars.clear();
    let (logits, used) = evaluate(&dr, &y[offset+2560..offset+5120])?;
    read=read.checked_add(used).ok_or("terminal decision read overflow")?;
    let d=imajev_runtime::decide_candidates(&options, &logits, 1.3051569717552742)?;
    let decision=ChoiceResult {value:d.value, probabilities:d.probabilities,
        unknown_probability:d.unknown_probability, abstained:d.abstained, raw_logits:d.raw_logits,
        instructions:ic_cdk::api::performance_counter(0)-decision_start,
        calibration_version:"p3-r2-s000291-authored".into()};
    r.step+=1;
    let state=encode(&r, &y)?;
    #[cfg(target_arch="wasm32")]
    let heap_pages=core::arch::wasm32::memory_size(0) as u64;
    #[cfg(not(target_arch="wasm32"))]
    let heap_pages=0;
    Ok(TerminalDecisionMeasurement {measurement:Measurement {state:state.into(),
        instructions:ic_cdk::api::performance_counter(0)-start, stable_read_bytes:read,
        heap_pages, stable_pages:ic_cdk::api::stable_size()}, decision})
}
#[ic_cdk::query]
fn decision_fast(
    state: Vec<u8>,
    options: Vec<String>,
) -> std::result::Result<ChoiceResult, String> {
    owner();
    let start = ic_cdk::api::performance_counter(0u32);
    let (mut r, x) = decode(&state)?;
    if r.op != "matmul"
        || !matches!(r.tensor.as_str(), "readout-f32" | "readout-int8")
        || r.dims != vec![1, 256, 2560]
        || x.len() != 2560
        || !(2..=7).contains(&options.len())
    {
        return Err("expected dedicated readout request/options".into());
    }
    r.dims = vec![1, options.len() + 1, 2560, 0];
    let (logits, _) = evaluate(&r, &x)?;
    let d = imajev_runtime::decide_candidates(&options, &logits, 1.3051569717552742)?;
    Ok(ChoiceResult {
        value: d.value,
        probabilities: d.probabilities,
        unknown_probability: d.unknown_probability,
        abstained: d.abstained,
        raw_logits: d.raw_logits,
        instructions: ic_cdk::api::performance_counter(0u32) - start,
        calibration_version: "p3-r2-s000291-authored".into(),
    })
}
#[derive(CandidType, Deserialize)]
struct ChoiceResult {
    value: Option<String>,
    probabilities: Vec<f32>,
    unknown_probability: f32,
    abstained: bool,
    raw_logits: Vec<f32>,
    instructions: u64,
    calibration_version: String,
}
#[ic_cdk::query]
fn decision(state: StateBytes, options: Vec<String>) -> std::result::Result<ChoiceResult, String> {
    owner();
    let (r, x) = decode(&state)?;
    if r.op != "matmul"
        || !matches!(r.tensor.as_str(), "readout-f32" | "readout-int8")
        || r.dims != vec![1, 256, 2560]
        || x.len() != 2560
    {
        return Err("expected dedicated readout request".into());
    }
    let measured = step(state.into())?;
    let (_, logits) = decode(&measured.state)?;
    let result = imajev_runtime::decide(&options, &logits, 1.3051569717552742)?;
    Ok(ChoiceResult {
        value: result.value,
        probabilities: result.probabilities,
        unknown_probability: result.unknown_probability,
        abstained: result.abstained,
        raw_logits: result.raw_logits,
        instructions: measured.instructions,
        calibration_version: "p3-r2-s000291-authored".into(),
    })
}
#[derive(CandidType, Deserialize)]
struct WeightCacheInfo {
    bytes: u64,
    names: Vec<String>,
    preparation_instructions: u64,
    rope_bytes: Option<u64>,
    activation_bytes: Option<u64>,
    paired_weight_bytes: Option<u64>,
}
fn cache_info(s: &Store, instructions: u64) -> WeightCacheInfo {
    #[cfg(feature="experimental-prepared-rope")]
    let rope_bytes=Some(imajev_runtime::rope_constant_bytes() as u64);
    #[cfg(not(feature="experimental-prepared-rope"))]
    let rope_bytes=None;
    #[cfg(feature="experimental-prepared-activation")]
    let activation_bytes=Some(imajev_runtime::prepared_activation::bytes() as u64);
    #[cfg(not(feature="experimental-prepared-activation"))]
    let activation_bytes=None;
    #[cfg(feature="experimental-prepared-output-pairs")]
    let paired_weight_bytes=Some(s.weight_cache.paired_weight_bytes());
    #[cfg(not(feature="experimental-prepared-output-pairs"))]
    let paired_weight_bytes=None;
    WeightCacheInfo { bytes: s.weight_cache.bytes(), names: s.weight_cache.names(), preparation_instructions: instructions, rope_bytes, activation_bytes, paired_weight_bytes }
}
/// Prepare only immutable weights. Upgrade drops this optional heap cache.
#[ic_cdk::update]
fn warm_weights(name: String) -> std::result::Result<WeightCacheInfo, String> {
    owner();
    if name.len() > 256 { return Err("weight name length".into()); }
    let start = ic_cdk::api::performance_counter(0);
    STORE.with(|s| {
        let mut s = s.borrow_mut();
        if !s.ready { return Err("not ready".into()); }
        let manifest = s.manifest.as_ref().ok_or("missing manifest")?;
        let tensor = manifest.tensors.iter().find(|t| t.name == name).ok_or("warm tensor")?.clone();
        let pack_bytes = manifest.bytes;
        #[cfg(feature = "experimental-strassen-prepared")]
        if tensor.dtype == "strassen-i8-v1" {
            use imajev_runtime::strassen_prepacked::PreparedStrassen;
            let source = name.strip_suffix(".strassen_i8").ok_or("prepared Strassen name")?;
            if s.weight_cache.strassen(source).is_none() {
                s.weight_cache.check_strassen_budget(PreparedStrassen::required_bytes(manifest, source)?)?;
                let value = PreparedStrassen::prepare(manifest, source, |offset, len| {
                    if offset.checked_add(len as u64).is_none_or(|end| end > pack_bytes) { return Err("prepared Strassen read range".into()); }
                    let mut bytes = vec![0; len];
                    ic_cdk::api::stable_read(offset, &mut bytes);
                    Ok(bytes)
                })?;
                s.weight_cache.insert_strassen(value)?;
            }
            return Ok(cache_info(&s, ic_cdk::api::performance_counter(0) - start));
        }
        s.weight_cache.check_insert(&tensor, pack_bytes)?;
        #[cfg(feature="experimental-prepared-rope")]
        imajev_runtime::prepare_rope_constants();
        #[cfg(feature="experimental-prepared-activation")]
        imajev_runtime::prepared_activation::prepare();
        if !s.weight_cache.contains(&tensor) {
            let mut bytes = vec![0; tensor.bytes as usize];
            ic_cdk::api::stable_read(tensor.offset, &mut bytes);
            s.weight_cache.insert(&tensor, bytes, pack_bytes)?;
        }
        Ok(cache_info(&s, ic_cdk::api::performance_counter(0) - start))
    })
}
#[ic_cdk::update]
fn clear_weight_cache() -> WeightCacheInfo {
    owner();
    #[cfg(feature="experimental-prepared-rope")]
    imajev_runtime::clear_rope_constants();
    #[cfg(feature="experimental-prepared-activation")]
    imajev_runtime::prepared_activation::clear();
    STORE.with(|s| { let mut s = s.borrow_mut(); s.weight_cache.clear(); cache_info(&s, 0) })
}
#[ic_cdk::query]
fn weight_cache_status() -> WeightCacheInfo {
    owner();
    STORE.with(|s| cache_info(&s.borrow(), 0))
}
#[ic_cdk::query]
fn status() -> (u64, bool) {
    owner();
    STORE.with(|s| (s.borrow().received, s.borrow().ready))
}
// Metadata survives upgrades separately from weight bytes. Restore only after compatible upgrade.
#[ic_cdk::pre_upgrade]
fn pre_upgrade() {
    owner_guardless_persist();
}
fn owner_guardless_persist() {
    STORE.with(|s| {
        let s = s.borrow();
        if let Some(m) = &s.manifest {
            let metadata =
                serde_json::to_vec(&(s.owner.unwrap().to_text(), m, s.received, s.ready)).unwrap();
            let end = (m.bytes + 65535) / 65536 * 65536;
            let needed = (metadata.len() as u64 + 16 + 65535) / 65536;
            let total = end / 65536 + needed;
            if total > ic_cdk::api::stable_size() {
                assert_ne!(
                    ic_cdk::api::stable_grow(total - ic_cdk::api::stable_size()),
                    u64::MAX
                )
            }
            ic_cdk::api::stable_write(end, &(metadata.len() as u64).to_le_bytes());
            ic_cdk::api::stable_write(end + 8, &metadata);
            ic_cdk::api::stable_write(total * 65536 - 8, &end.to_le_bytes());
        }
    })
}
#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let size = ic_cdk::api::stable_size() * 65536;
    assert!(size >= 8, "missing upgrade metadata");
    let mut pointer = [0; 8];
    ic_cdk::api::stable_read(size - 8, &mut pointer);
    let end = u64::from_le_bytes(pointer);
    assert!(end + 8 < size);
    let mut length = [0; 8];
    ic_cdk::api::stable_read(end, &mut length);
    let n = u64::from_le_bytes(length);
    assert!(n < 2_000_000 && end + 8 + n <= size - 8);
    let mut bytes = vec![0; n as usize];
    ic_cdk::api::stable_read(end + 8, &mut bytes);
    let (owner, manifest, received, ready): (String, Manifest, u64, bool) =
        serde_json::from_slice(&bytes).unwrap();
    STORE.with(|s| {
        *s.borrow_mut() = Store {
            owner: Some(Principal::from_text(owner).unwrap()),
            manifest: Some(manifest),
            received: if ready { received } else { 0 },
            ready,
            digest: Sha256::new(),
            chunks: Default::default(),
            hashed: if ready { received } else { 0 },
            weight_cache: Default::default(),
        }
    });
}
ic_cdk::export_candid!();
pub fn get_candid_pointer_for_tests() -> String {
    __export_service()
}

#[cfg(all(test, feature = "experimental-byte-buffer"))]
mod state_bytes_tests {
    use super::*;
    #[test]
    fn byte_buffer_keeps_candid_blob_type_and_wire_bytes() {
        assert_eq!(<StateBytes as CandidType>::_ty(), <Vec<u8> as CandidType>::_ty());
        for n in [0, 1, 256, 900_000] {
            let bytes: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
            let old = candid::encode_args((bytes.clone(),)).unwrap();
            let new = candid::encode_args((StateBytes::from(bytes.clone()),)).unwrap();
            assert_eq!(old, new);
            let (decoded,): (StateBytes,) = candid::decode_args(&old).unwrap();
            assert_eq!(decoded.as_slice(), bytes);
        }
    }
}

#[cfg(test)]
mod terminal_decision_tests {
    use super::*;
    fn request() -> imajev_runtime::Request {
        serde_json::from_value(serde_json::json!({"version":2,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,
        "op":"terminal_attention_mlp_integer","tensor":"model.language_model.layers.31.self_attn.q_proj.weight","dims":[87,45],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap()
    }
    #[test]
    fn tail_decision_contract_checks_feature_layer_and_token_bound() {
        let mut r=request();r.op="terminal_tail_integer".into();r.tensor="model.language_model.layers.30.post_attention_layernorm.weight".into();
        let options=vec!["yes".into(),"no".into()];
        assert_eq!(validate_terminal_decision(&r,&options).is_ok(),cfg!(feature="experimental-terminal-tail"));
        r.dims[0]=90;assert!(validate_terminal_decision(&r,&options).is_err());
        r.dims[0]=87;r.tensor="model.language_model.layers.31.self_attn.q_proj.weight".into();assert!(validate_terminal_decision(&r,&options).is_err());
    }
    #[test]
    fn stream_decision_checks_progress_scope_and_options() {
        if cfg!(feature="experimental-mlp-attention-finish") {
            assert!(cfg!(feature="experimental-projection-reuse"), "bridge requests require the typed query decoder");
        }
        let mut r=request();r.op="mlp_stream_complete_terminal".into();
        r.tensor="model.language_model.layers.30.post_attention_layernorm.weight".into();
        r.encoding="mlp-attention-finish-exact-v1".into();r.dims=vec![87,45,2560];
        r.scalars=vec![2.,1e-6];r.aux=vec!["model.language_model.layers.31.input_layernorm.weight".into()];
        let options=vec!["yes".into(),"no".into()];
        assert_eq!(validate_terminal_decision(&r,&options).is_ok(),cfg!(feature="experimental-terminal-stream"));
        for dims in [vec![],vec![87,45],vec![0,45,2560],vec![90,45,2560],vec![87,133,2560],vec![87,45,0],vec![87,45,9216],vec![87,45,1]] {
            let mut invalid=r.clone();invalid.dims=dims;assert!(validate_terminal_decision(&invalid,&options).is_err());
        }
        for field in ["tensor","encoding","aux","scalars","step"] {
            let mut invalid=r.clone();match field {
                "tensor"=>invalid.tensor="model.language_model.layers.26.post_attention_layernorm.weight".into(),
                "encoding"=>invalid.encoding="bf16-block256-exact-v1".into(),
                "aux"=>invalid.aux.clear(),"scalars"=>invalid.scalars[1]=f32::NAN,
                _=>invalid.step=u64::MAX,
            };assert!(validate_terminal_decision(&invalid,&options).is_err());
        }
        for options in [vec![],vec!["yes".into()],vec!["yes".into(),"yes".into()],vec!["__unknown__".into(),"no".into()]] {
            assert!(validate_terminal_decision(&r,&options).is_err());
        }
    }
    #[test]
    fn fused_decision_contract_rejects_wrong_stage_and_options_before_inference() {
        let r=request();let options=vec!["yes".into(),"no".into()];
        assert!(validate_terminal_decision(&r,&options).is_ok());
        for dims in [vec![],vec![0,45],vec![133,0],vec![1,512],vec![usize::MAX,0]] {
            let mut invalid=r.clone();invalid.dims=dims;assert!(validate_terminal_decision(&invalid,&options).is_err());
        }
        for field in ["op","tensor","encoding","aux","scalars","step"] {
            let mut invalid=r.clone();match field {
                "op"=>invalid.op="attention_full_integer".into(),
                "tensor"=>invalid.tensor="model.language_model.layers.3.self_attn.q_proj.weight".into(),
                "encoding"=>invalid.encoding="int8-block256-v1".into(),
                "aux"=>invalid.aux.push("unexpected".into()),"scalars"=>invalid.scalars.push(1.),
                _=>invalid.step=u64::MAX,
            };assert!(validate_terminal_decision(&invalid,&options).is_err());
        }
        for options in [vec![],vec!["yes".into()],vec!["yes".into(),"yes".into()],vec!["".into(),"no".into()],vec!["__unknown__".into(),"no".into()],vec!["x".repeat(129),"no".into()],vec!["yes".into();8]] {
            assert!(validate_terminal_decision(&r,&options).is_err());
        }
        assert!(__export_service().contains("terminal_step_decision"));
    }
}
