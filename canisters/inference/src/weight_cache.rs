//! Immutable, bounded fixed-weight preparation; no inference state is retained.
#[cfg(all(feature = "experimental-strassen-packed-reduction", not(feature = "experimental-strassen-prepared")))]
compile_error!("Packed Strassen requires canister preparation support");
use imajev_runtime::{Tensor, prepared_weights::{PreparedF32, WeightBuffer}};
use std::collections::BTreeMap;
const TENSOR_LIMIT: u64 = 128 * 1024 * 1024;
#[cfg(not(feature = "experimental-full-weight-cache"))]
pub const CACHE_LIMIT: u64 = 256 * 1024 * 1024;
#[cfg(feature = "experimental-full-weight-cache")]
pub const CACHE_LIMIT: u64 = 4_080_000_000; // Dense INT8 + original F32; leave query scratch below 4GiB.
struct Entry {
    name: String,
    data: Data,
}
enum Data { Bytes(Vec<u8>), F32(PreparedF32), #[cfg(feature="experimental-prepared-output-pairs")] Pairs(imajev_runtime::output_pairs::PreparedPairs) }
impl Data {
    fn bytes_len(&self) -> usize { match self { Self::Bytes(b) => b.len(), Self::F32(v) => v.len() * 4, #[cfg(feature="experimental-prepared-output-pairs")] Self::Pairs(v)=>v.bytes() } }
}
pub enum ReadBuffer<'a> { Bytes(&'a [u8]), F32(PreparedF32), Owned(Vec<u8>), #[cfg(feature="experimental-prepared-output-pairs")] Pairs(imajev_runtime::output_pairs::OriginalView) }
impl AsRef<[u8]> for ReadBuffer<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Bytes(b) => b, Self::Owned(b) => b,
            #[cfg(feature="experimental-prepared-output-pairs")]
            Self::Pairs(v)=>v.as_ref(),
            Self::F32(v) => {
                // The selected Wasm/native targets are little-endian. Reading
                // initialized f32 as bytes is valid for all bit patterns/alignment1.
                #[cfg(target_endian = "big")]
                compile_error!("Fixed F32 byte cache requires little-endian target");
                unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
            }
        }
    }
}
impl WeightBuffer for ReadBuffer<'_> {
    #[cfg(feature="experimental-prepared-output-pairs")]
    fn prepared_output_pairs(&self)->Option<imajev_runtime::output_pairs::PackedView> {
        match self {Self::Pairs(v)=>v.packed(),_=>None}
    }
    fn prepared_f32(&self) -> Option<PreparedF32> {
        match self { Self::F32(v) => Some(v.clone()), #[cfg(feature="experimental-prepared-output-pairs")] Self::Pairs(v)=>v.prepared_scales(), _ => None }
    }
}
#[derive(Default)]
pub struct WeightCache {
    entries: BTreeMap<u64, Entry>,
    total: u64,
    #[cfg(feature = "experimental-strassen-prepared")]
    strassen: BTreeMap<String, imajev_runtime::strassen_prepacked::PreparedStrassen>,
}
impl WeightCache {
    pub fn bytes(&self) -> u64 { self.total }
    #[cfg(feature="experimental-prepared-output-pairs")]
    pub fn paired_weight_bytes(&self)->u64 {self.entries.values().map(|e|match &e.data {Data::Pairs(v)=>v.weight_bytes()as u64,_=>0}).sum()}
    pub fn names(&self) -> Vec<String> {
        let names = self.entries.values().map(|entry| entry.name.clone());
        #[cfg(feature = "experimental-strassen-prepared")]
        let names = names.chain(self.strassen.keys().map(|name| format!("{name}.strassen_i8")));
        names.collect()
    }
    pub fn contains(&self, t: &Tensor) -> bool {
        self.entries.get(&t.offset).is_some_and(|entry| entry.name == t.name && entry.data.bytes_len() as u64 == t.bytes)
    }
    pub fn check_insert(&self, t: &Tensor, pack_bytes: u64) -> Result<(), String> {
        let count = t.rows.checked_mul(t.cols).ok_or("cache tensor overflow")?;
        let expected = match t.dtype.as_str() {
            "int8" => count.checked_add(t.rows.checked_mul(4).ok_or("cache scale overflow")?),
            "f32" => count.checked_mul(4),
            _ => None,
        }.ok_or("cache tensor precision/overflow")?;
        let end = t.offset.checked_add(t.bytes).ok_or("cache offset overflow")?;
        if t.rows == 0 || t.cols == 0 || t.bytes != expected as u64 || t.bytes > TENSOR_LIMIT || end > pack_bytes {
            return Err("cache tensor shape/range".into());
        }
        if self.contains(t) { return Ok(()); }
        if self.total.checked_add(t.bytes).is_none_or(|total| total > CACHE_LIMIT) {
            return Err("fixed weight cache limit".into());
        }
        if self.entries.range(..=t.offset).next_back().is_some_and(|(offset, entry)| offset + entry.data.bytes_len() as u64 > t.offset)
            || self.entries.range(t.offset..).next().is_some_and(|(offset, _)| *offset < end) {
            return Err("overlapping cached tensor".into());
        }
        Ok(())
    }
    pub fn insert(&mut self, t: &Tensor, bytes: Vec<u8>, pack_bytes: u64) -> Result<(), String> {
        self.check_insert(t, pack_bytes)?;
        if bytes.len() as u64 != t.bytes { return Err("cache read size".into()); }
        if self.contains(t) { return Ok(()); }
        let data = if t.dtype == "f32" {
            {
                #[cfg(feature="experimental-f32-output-reuse")]
                if (t.name.ends_with(".lora_B.weight") && t.cols==64 || cfg!(feature="experimental-f32-output-generic") && t.name.ends_with(".lora_A.weight") && t.cols%64==0 && t.cols<=9216) && t.rows%32==0 {
                    Data::F32(PreparedF32::from_le_bytes_output(&bytes,t.rows,t.cols)?)
                } else {Data::F32(PreparedF32::from_le_bytes(&bytes)?)}
                #[cfg(not(feature="experimental-f32-output-reuse"))]
                {Data::F32(PreparedF32::from_le_bytes(&bytes)?)}
            }
        } else {
            let count = t.rows * t.cols; // Checked before reading or indexing.
            if bytes[count..].chunks_exact(4).any(|v| {
                let scale = f32::from_le_bytes(v.try_into().unwrap());
                !scale.is_finite() || scale <= 0.
            }) { return Err("cache weight scales".into()); }
            #[cfg(feature="experimental-prepared-output-pairs")]
            if t.rows>=(if cfg!(feature="experimental-paired-only"){32}else{512}) && t.rows%32==0 && t.cols%256==0 && count<=30_000_000 {
                Data::Pairs(imajev_runtime::output_pairs::PreparedPairs::from_le_bytes(&bytes,t.rows,t.cols)?)
            }else {Data::Bytes(bytes)}
            #[cfg(not(feature="experimental-prepared-output-pairs"))]
            {Data::Bytes(bytes)}
        };
        self.total += t.bytes;
        self.entries.insert(t.offset, Entry { name: t.name.clone(), data });
        Ok(())
    }
    pub fn read(&self, offset: u64, len: usize) -> Option<ReadBuffer<'_>> {
        let (start, entry) = self.entries.range(..=offset).next_back()?;
        let begin = usize::try_from(offset.checked_sub(*start)?).ok()?;
        match &entry.data {
            #[cfg(feature="experimental-prepared-output-pairs")]
            Data::Pairs(v)=>Some(ReadBuffer::Pairs(v.original_view(begin,len)?)),
            Data::Bytes(b) => Some(ReadBuffer::Bytes(b.get(begin..begin.checked_add(len)?)?)),
            Data::F32(v) => {
                if begin % 4 != 0 || len % 4 != 0 { return None; }
                Some(ReadBuffer::F32(v.slice(begin / 4, len / 4)?))
            }
        }
    }
    #[cfg(feature = "experimental-strassen-prepared")]
    pub fn strassen(&self, name: &str) -> Option<&imajev_runtime::strassen_prepacked::PreparedStrassen> { self.strassen.get(name) }
    #[cfg(feature = "experimental-strassen-prepared")]
    pub fn check_strassen_budget(&self, bytes: u64) -> Result<(), String> {
        if bytes > TENSOR_LIMIT || self.total.checked_add(bytes).is_none_or(|v| v > CACHE_LIMIT) {
            return Err("prepared Strassen cache limit".into());
        }
        Ok(())
    }
    #[cfg(feature = "experimental-strassen-prepared")]
    pub fn insert_strassen(&mut self, value: imajev_runtime::strassen_prepacked::PreparedStrassen) -> Result<(), String> {
        if self.strassen.contains_key(value.source()) { return Ok(()); }
        self.check_strassen_budget(value.bytes())?;
        self.total += value.bytes();
        self.strassen.insert(value.source().into(), value);
        Ok(())
    }
    pub fn clear(&mut self) {
        self.entries.clear(); self.total = 0;
        #[cfg(feature = "experimental-strassen-prepared")]
        self.strassen.clear();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature="experimental-f32-output-reuse")]
    #[test]
    fn output_f32_cache_keeps_payload_size_and_original_nonzero_ranges() {
        let rows=256;let cols=64;
        let values:Vec<f32>=(0..rows*cols).map(|i|[-0.,0.,0.00123,-0.1234567,f32::MIN_POSITIVE][i%5]).collect();
        let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
        let tensor=Tensor{name:"model.layers.0.lora_B.weight".into(),offset:1024,rows,cols,dtype:"f32".into(),bytes:bytes.len()as u64};
        let mut cache=WeightCache::default();cache.insert(&tensor,bytes.clone(),tensor.offset+tensor.bytes).unwrap();
        assert_eq!(cache.bytes(),tensor.bytes);
        for (start,count) in [(0,rows*cols),(3,513),(8*cols,32*cols),(rows*cols-3,3)] {
            let view=cache.read(tensor.offset+start as u64*4,count*4).unwrap();
            assert_eq!(view.prepared_f32().unwrap().len(),count);
            assert_eq!(view.as_ref(),&bytes[start*4..(start+count)*4]);
        }
        cache.insert(&tensor,bytes,tensor.offset+tensor.bytes).unwrap();assert_eq!(cache.bytes(),tensor.bytes);
        cache.clear();assert_eq!(cache.bytes(),0);
    }
    #[cfg(feature="experimental-paired-only")]
    #[test]
    fn small_gate_weights_are_prepared_once_without_payload_growth() {
        let rows=32;let cols=2560;
        let mut bytes:Vec<u8>=(0..rows*cols).map(|i|(i%256)as u8).collect();
        for _ in 0..rows {bytes.extend(0.0123f32.to_le_bytes());}
        let t=Tensor{name:"small-gate".into(),offset:0,rows,cols,dtype:"int8".into(),bytes:bytes.len()as u64};
        let mut cache=WeightCache::default();cache.insert(&t,bytes.clone(),t.bytes).unwrap();
        assert_eq!(cache.bytes(),t.bytes);
        assert_eq!(cache.paired_weight_bytes(),(rows*cols)as u64);
        let buffer=cache.read(0,rows*cols).unwrap();
        assert_eq!(buffer.prepared_output_pairs().unwrap().rows(),rows);
        assert_eq!(buffer.as_ref(),&bytes[..rows*cols]);
        assert_eq!(cache.read((rows*cols)as u64,rows*4).unwrap().as_ref(),&bytes[rows*cols..]);
    }
    #[cfg(feature="experimental-prepared-output-pairs")]
    #[test]
    fn packed_fixed_cache_keeps_original_reads_and_payload_size() {
        let rows=512;let cols=256;
        let mut bytes:Vec<u8>=(0..rows*cols).map(|i|(i%256)as u8).collect();
        for _ in 0..rows {bytes.extend(0.0123f32.to_le_bytes());}
        let t=Tensor{name:"paired".into(),offset:1024,rows,cols,dtype:"int8".into(),bytes:bytes.len()as u64};
        let mut cache=WeightCache::default();cache.insert(&t,bytes.clone(),t.offset+t.bytes).unwrap();
        assert_eq!(cache.bytes(),bytes.len()as u64);assert_eq!(cache.paired_weight_bytes(),(rows*cols)as u64);
        let v=cache.read(1024,rows*cols).unwrap();assert_eq!(v.prepared_output_pairs().unwrap().rows(),rows);
        if let ReadBuffer::Pairs(p)=&v {assert!(!p.original_was_decoded());}else{panic!("packed cache missing");}
        assert_eq!(v.as_ref(),&bytes[..rows*cols]);drop(v);
        for (s,n) in [(1,31),(255,513),(rows*cols-3,128),(rows*cols,rows*4)] {
            assert_eq!(cache.read(1024+s as u64,n).unwrap().as_ref(),&bytes[s..s+n]);
        }
        cache.clear();assert_eq!(cache.paired_weight_bytes(),0);
    }
    fn fixture(offset: u64) -> (Tensor, Vec<u8>) {
        let mut bytes = vec![0x80; 8 * 256];
        for _ in 0..8 { bytes.extend(0.01f32.to_le_bytes()); }
        (Tensor { name: format!("w{offset}"), offset, rows: 8, cols: 256, dtype: "int8".into(), bytes: bytes.len() as u64 }, bytes)
    }
    #[test]
    fn immutable_ranges_cover_weights_scales_and_clear() {
        let (t, bytes) = fixture(1024); let expected = bytes.clone();
        let mut cache = WeightCache::default(); cache.insert(&t, bytes, 8192).unwrap();
        assert_eq!(cache.bytes(), t.bytes);
        assert_eq!(cache.read(1031, 19).unwrap().as_ref(), &expected[7..26]);
        assert_eq!(cache.read(1024 + 2048, 32).unwrap().as_ref(), &expected[2048..]);
        assert!(cache.read(1023, 1).is_none());
        assert!(cache.read(1024 + t.bytes - 1, 2).is_none());
        assert!(cache.read(u64::MAX, usize::MAX).is_none());
        cache.insert(&t, expected, 8192).unwrap(); assert_eq!(cache.bytes(), t.bytes);
        cache.clear(); assert_eq!(cache.bytes(), 0); assert!(cache.read(1024, 1).is_none());
    }
    #[test]
    fn invalid_fixed_weights_and_overlap_are_rejected() {
        let (t, bytes) = fixture(1024); let mut cache = WeightCache::default();
        for bad in [f32::NAN, f32::INFINITY, 0., -1.] {
            let mut corrupt = bytes.clone(); corrupt[2048..2052].copy_from_slice(&bad.to_le_bytes());
            assert!(cache.insert(&t, corrupt, 8192).is_err()); assert_eq!(cache.bytes(), 0);
        }
        assert!(cache.insert(&t, bytes[..2079].to_vec(), 8192).is_err());
        assert!(cache.check_insert(&t, 1024).is_err());
        let mut malformed = t.clone(); malformed.cols = usize::MAX;
        assert!(cache.check_insert(&malformed, u64::MAX).is_err());
        malformed = t.clone(); malformed.offset = u64::MAX;
        assert!(cache.check_insert(&malformed, u64::MAX).is_err());
        malformed = t.clone(); malformed.dtype = "bf16".into();
        assert!(cache.check_insert(&malformed, 8192).is_err());
        cache.insert(&t, bytes, 8192).unwrap();
        let (overlap, _) = fixture(2048); assert!(cache.check_insert(&overlap, 8192).is_err());
    }
    #[test]
    fn prepared_f32_reads_borrow_exact_nonzero_tiles_and_reject_nonfinite() {
        let values = [0., -0., f32::from_bits(1), f32::MAX, -1., 2.];
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = Tensor { name: "adapter".into(), offset: 1001, rows: 2, cols: 3, dtype: "f32".into(), bytes: 24 };
        let mut cache = WeightCache::default();
        cache.insert(&t, bytes.clone(), 2048).unwrap();
        let full = cache.read(1001, 24).unwrap();
        let tile = cache.read(1013, 12).unwrap();
        assert_eq!(tile.as_ref(), &bytes[12..]);
        let prepared = tile.prepared_f32().unwrap();
        assert_eq!(prepared.as_ptr(), full.prepared_f32().unwrap()[3..].as_ptr());
        assert!(cache.read(1002, 4).is_none());
        assert!(cache.read(1001, 3).is_none());
        cache.clear();
        assert_eq!(prepared[0].to_bits(), f32::MAX.to_bits()); // Rc keeps a valid immutable view.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut corrupt = bytes.clone(); corrupt[4..8].copy_from_slice(&bad.to_le_bytes());
            assert!(cache.insert(&t, corrupt, 2048).is_err());
            assert_eq!(cache.bytes(), 0);
        }
    }
}
