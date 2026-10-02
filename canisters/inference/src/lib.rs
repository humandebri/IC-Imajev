use candid::{CandidType, Principal};
use imajev_runtime::{decode, encode, Manifest};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
#[derive(Default)]
struct Store {
    owner: Option<Principal>,
    manifest: Option<Manifest>,
    received: u64,
    ready: bool,
    digest: Sha256,
    chunks: std::collections::BTreeMap<u64, Vec<u8>>,
    hashed: u64,
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
#[derive(CandidType, Deserialize)]
struct Measurement {
    state: Vec<u8>,
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
fn step(state: Vec<u8>) -> std::result::Result<Measurement, String> {
    owner();
    let start = ic_cdk::api::performance_counter(0u32);
    let (mut r, x) = decode(&state)?;
    let (y, read) = evaluate(&r, &x)?;
    r.step = r.step.checked_add(1).ok_or("progress overflow")?;
    let state = encode(&r, &y)?;
    #[cfg(target_arch = "wasm32")]
    let heap_pages = core::arch::wasm32::memory_size(0) as u64;
    #[cfg(not(target_arch = "wasm32"))]
    let heap_pages = 0;
    Ok(Measurement {
        state,
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
fn profile_step(state: Vec<u8>) -> std::result::Result<ProfileMeasurement, String> {
    owner();
    if !cfg!(feature = "instruction-profile") {
        return Err("instruction-profile feature is disabled".into());
    }
    imajev_runtime::profile::start(|| ic_cdk::api::performance_counter(0));
    let result = (|| {
        let start = ic_cdk::api::performance_counter(0);
        let (mut r, x) = imajev_runtime::profile::measure("wire_decode", || decode(&state))?;
        let (y, read) =
            imajev_runtime::profile::measure("evaluate_inclusive", || evaluate(&r, &x))?;
        r.step = r.step.checked_add(1).ok_or("progress overflow")?;
        let state = imajev_runtime::profile::measure("wire_encode", || encode(&r, &y))?;
        #[cfg(target_arch = "wasm32")]
        let heap_pages = core::arch::wasm32::memory_size(0) as u64;
        #[cfg(not(target_arch = "wasm32"))]
        let heap_pages = 0;
        Ok(Measurement {
            state,
            instructions: ic_cdk::api::performance_counter(0) - start,
            stable_read_bytes: read,
            heap_pages,
            stable_pages: ic_cdk::api::stable_size(),
        })
    })();
    let spans = imajev_runtime::profile::finish();
    result.map(|measurement| ProfileMeasurement { measurement, spans })
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
        imajev_runtime::evaluate_with_reader(r, x, m, |offset, len| {
            let mut bytes = vec![0; len];
            ic_cdk::api::stable_read(offset, &mut bytes);
            Ok(bytes)
        })
    })
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
fn decision(state: Vec<u8>, options: Vec<String>) -> std::result::Result<ChoiceResult, String> {
    let (r, x) = decode(&state)?;
    if r.op != "matmul"
        || !matches!(r.tensor.as_str(), "readout-f32" | "readout-int8")
        || r.dims != vec![1, 256, 2560]
        || x.len() != 2560
    {
        return Err("expected dedicated readout request".into());
    }
    let measured = step(state)?;
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
        }
    });
}
ic_cdk::export_candid!();
pub fn get_candid_pointer_for_tests() -> String {
    __export_service()
}
