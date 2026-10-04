//! Lossless block256 BF16/F32 transport. A full block needs at least one F32 exception.
use crate::{bf16_codec, Result, MAX_FLOATS};
pub fn append(b: &mut Vec<u8>, x: &[f32]) -> Result<()> {
    b.extend_from_slice(&(x.len() as u32).to_le_bytes());
    let bitmap = b.len();
    b.resize(bitmap + x.len().div_ceil(256).div_ceil(8), 0);
    // Classification and finite validation happen once per block. Reuse the
    // flags for both sizing and encoding; do not scan the entire input first.
    let mut all=true;
    let mut payload_bytes=0usize;
    for(i,chunk)in x.chunks(256).enumerate(){
        let full=!bf16_codec::classify_finite(chunk)?;
        if full{all=false;b[bitmap+i/8]|=1<<(i%8);}
        payload_bytes+=chunk.len()*if full{4}else{2};
    }
    if b.len()+payload_bytes+32>2_000_000{return Err("block codec size".into());}
    if all{let start=b.len();b.resize(start+payload_bytes,0);bf16_codec::pack(x,&mut b[start..]);return Ok(());}
    for(i,chunk)in x.chunks(256).enumerate(){
        let full=b[bitmap+i/8]&(1<<(i%8))!=0;
        let start=b.len();let width=if full{4}else{2};b.resize(start+chunk.len()*width,0);
        if full{for(v,bytes)in chunk.iter().zip(b[start..].chunks_exact_mut(4)){bytes.copy_from_slice(&v.to_le_bytes());}}
        else{bf16_codec::pack(chunk,&mut b[start..]);}
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
    let mut values = Vec::<f32>::with_capacity(count);
    let mut cursor = 4 + size;
    if flags.iter().all(|v| *v == 0) {
        bf16_codec::unpack_finite_uninit(&payload[cursor..], &mut values.spare_capacity_mut()[..count])?;
        // SAFETY: decoder wrote all count elements and verified finiteness.
        unsafe{values.set_len(count);}
        return Ok(values);
    }
    for (i, chunk) in values.spare_capacity_mut()[..count].chunks_mut(256).enumerate() {
        let full = flags[i / 8] & (1 << (i % 8)) != 0;
        let len = chunk.len() * if full { 4 } else { 2 };
        let bytes = &payload[cursor..cursor + len];
        if full {
            if bf16_codec::unpack_f32_finite_uninit(bytes,chunk)? {
                return Err("block codec noncanonical".into());
            }
        } else {
            bf16_codec::unpack_finite_uninit(bytes, chunk)?;
        }
        cursor += len;
    }
    // SAFETY: every disjoint chunk initialized exactly its complete span.
    // Nonfinite/noncanonical errors return before exposing any Vec element.
    unsafe{values.set_len(count);}
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
    fn checked_codec_matches_scalar_format_and_rejects_late_nonfinite(){
        for n in [0usize,1,7,8,9,255,256,257,511,512,513,2049]{
            let x:Vec<f32>=(0..n).map(|i|if (i/256)%2==0 {f32::from_bits((0x8000+(i as u32%0x7f80))<<16)}else{f32::from_bits(0x3f800001+(i as u32%0xffff))}).collect();
            let mut expected=(n as u32).to_le_bytes().to_vec();expected.resize(4+n.div_ceil(256).div_ceil(8),0);
            for(i,c)in x.chunks(256).enumerate(){let full=c.iter().any(|v|v.to_bits()&65535!=0);if full{expected[4+i/8]|=1<<(i%8);}for v in c{if full{expected.extend(v.to_bits().to_le_bytes());}else{expected.extend(((v.to_bits()>>16)as u16).to_le_bytes());}}}
            let mut actual=vec![];append(&mut actual,&x).unwrap();assert_eq!(actual,expected);
            let y=decode(&expected).unwrap();assert!(x.iter().zip(y).all(|(a,b)|a.to_bits()==b.to_bits()));
        }
        for bits in [0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xff800001,0xffffffff]{for at in [0,7,8,255,256,512]{
            let mut x=vec![1.;513];x[at]=f32::from_bits(bits);assert!(append(&mut vec![],&x).is_err());
            // A complete, canonical F32 block with a nonfinite late value.
            let mut frame=513u32.to_le_bytes().to_vec();frame.push(7);for v in &x{frame.extend(v.to_bits().to_le_bytes());}
            assert!(decode(&frame).is_err());
            // A nonfinite BF16 value must also fail the all-BF16 fast path.
            let mut b=513u32.to_le_bytes().to_vec();b.push(0);for v in &x{b.extend(((v.to_bits()>>16)as u16).to_le_bytes());}assert!(decode(&b).is_err());
        }}
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
