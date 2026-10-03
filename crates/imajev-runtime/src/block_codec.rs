//! Lossless block256 BF16/F32 transport. A full block needs at least one F32 exception.
use crate::{bf16_codec, Result, MAX_FLOATS};
pub fn append(b: &mut Vec<u8>, x: &[f32]) -> Result<()> {
    b.extend_from_slice(&(x.len() as u32).to_le_bytes());
    let bitmap = b.len();
    b.resize(bitmap + x.len().div_ceil(256).div_ceil(8), 0);
    if bf16_codec::all_bf16(x) {
        let start = b.len();
        if start + x.len() * 2 + 32 > 2_000_000 {
            return Err("block codec size".into());
        }
        b.resize(start + x.len() * 2, 0);
        bf16_codec::pack(x, &mut b[start..]);
        return Ok(());
    }
    for (i, chunk) in x.chunks(256).enumerate() {
        let full = !bf16_codec::all_bf16(chunk);
        let start = b.len();
        let width = if full { 4 } else { 2 };
        if start + chunk.len() * width + 32 > 2_000_000 {
            return Err("block codec size".into());
        }
        b.resize(start + chunk.len() * width, 0);
        if full {
            b[bitmap + i / 8] |= 1 << (i % 8);
            for (v, bytes) in chunk.iter().zip(b[start..].chunks_exact_mut(4)) {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        } else {
            bf16_codec::pack(chunk, &mut b[start..]);
        }
    }
    Ok(())
}
pub fn decode(payload: &[u8]) -> Result<Vec<f32>> {
    if payload.len() < 4 {
        return Err("block codec count".into());
    }
    let count = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    if count > MAX_FLOATS {
        return Err("block codec count bound".into());
    }
    let blocks = count.div_ceil(256);
    let size = blocks.div_ceil(8);
    if payload.len() < 4 + size {
        return Err("block codec bitmap".into());
    }
    let flags = &payload[4..4 + size];
    if blocks % 8 != 0 && flags.last().is_some_and(|v| v >> (blocks % 8) != 0) {
        return Err("block codec padding".into());
    }
    let mut expected = 4 + size;
    for i in 0..blocks {
        expected += (count - i * 256).min(256)
            * if flags[i / 8] & (1 << (i % 8)) != 0 {
                4
            } else {
                2
            };
    }
    if payload.len() != expected {
        return Err("block codec length".into());
    }
    let mut values = vec![0.; count];
    let mut cursor = 4 + size;
    if flags.iter().all(|v| *v == 0) {
        bf16_codec::unpack(&payload[cursor..], &mut values);
        return Ok(values);
    }
    for (i, chunk) in values.chunks_mut(256).enumerate() {
        let full = flags[i / 8] & (1 << (i % 8)) != 0;
        let len = chunk.len() * if full { 4 } else { 2 };
        let bytes = &payload[cursor..cursor + len];
        if full {
            for (v, b) in chunk.iter_mut().zip(bytes.chunks_exact(4)) {
                *v = f32::from_le_bytes(b.try_into().unwrap());
            }
            if bf16_codec::all_bf16(chunk) {
                return Err("block codec noncanonical".into());
            }
        } else {
            bf16_codec::unpack(bytes, chunk);
        }
        cursor += len;
    }
    Ok(values)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocks_preserve_bits_and_partial_tails() {
        for n in [0, 1, 255, 256, 257, 511, 512, 513, 2049] {
            let mut x = vec![-0.; n];
            if n > 0 {
                x[0] = f32::from_bits(1);
            }
            if n > 256 {
                x[256] = 1.0000001;
            }
            let mut b = vec![];
            append(&mut b, &x).unwrap();
            let y = decode(&b).unwrap();
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            b.push(0);
            assert!(decode(&b).is_err());
        }
    }
    #[test]
    fn redundant_full_blocks_and_padding_are_rejected() {
        let mut b = 1u32.to_le_bytes().to_vec();
        b.push(1);
        b.extend(1f32.to_le_bytes());
        assert!(decode(&b).is_err());
        b[4] = 128;
        assert!(decode(&b).is_err());
        assert!(decode(&900001u32.to_le_bytes()).is_err());
    }
}
