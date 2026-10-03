//! Client-held exact quantized inputs and original F32 LoRA-A products.
//! Capture fuses preparation into the first output tile; reuse retains no heap state.
use crate::{int8_kernel, prepared_weights::WeightBuffer, Manifest, Request, Result};

pub(super) fn evaluate<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer,
{
    if !crate::lossless_encoding(&r.encoding) || r.dims.len() != if r.encoding == crate::projection_codec::NAME {5} else {4} || r.aux.len() != 2
        || r.scalars.len() != 1 || !r.scalars[0].is_finite() {
        return Err("projection reuse metadata".into());
    }
    let (n, rows, cols, start) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3]);
    if n == 0 || n > 512 || rows == 0 || rows % 8 != 0 || cols == 0 || cols % 256 != 0
        || rows.checked_mul(cols).is_none_or(|v| v > 30_000_000)
        || n.checked_mul(rows).is_none_or(|v| v > crate::MAX_FLOATS)
        || n.checked_mul(cols).is_none_or(|v| v > crate::MAX_FLOATS) {
        return Err("projection reuse bounds".into());
    }
    let a = m.tensors.iter().find(|t| t.name == r.aux[0]).ok_or("projection reuse A")?;
    let b = m.tensors.iter().find(|t| t.name == r.aux[1]).ok_or("projection reuse B")?;
    let base = m.tensors.iter().find(|t| t.name == r.tensor).ok_or("projection reuse base")?;
    if (r.dims.len()==5 && r.dims[4]!=a.rows) || a.rows == 0 || a.rows > 256 || a.cols != cols || b.cols != a.rows
        || base.dtype != "int8" || base.cols != cols
        || start.checked_add(rows).is_none_or(|v| v > base.rows || v > b.rows)
        || n as u64 * (rows as u64 * cols as u64 + a.rows as u64 * (rows + cols) as u64) > 4_000_000_000 {
        return Err("projection reuse shape/work".into());
    }
    let q_len = n * cols;
    let sx_len = n * (cols / 256);
    let ax_len = n * a.rows;
    let state_len = q_len + sx_len + ax_len;
    if state_len > crate::MAX_FLOATS { return Err("projection reuse state size".into()); }
    let capture = r.op == "lora_integer_capture";
    if x.len() != if capture { q_len } else { state_len }
        || !x.iter().all(|v| v.is_finite())
        || (capture && n * rows + state_len > crate::MAX_FLOATS) {
        return Err("projection reuse frame bounds".into());
    }
    let mut ordinary = r.clone();
    ordinary.op = "lora_integer".into();
    ordinary.dims.truncate(4);
    if capture {
        let q = crate::profile::measure("activation_quantize", || int8_kernel::quantize_rows(x, n, cols))?;
        let mut ar = ordinary.clone();
        ar.op = "matmul".into(); ar.dims = vec![n, a.rows, cols];
        let (aw, ra) = crate::load_prepared_weight(a, &ar, &mut *read)?;
        let ax = crate::profile::measure("lora_matmul_A", || crate::matrix_loaded(x, &aw, n, a.rows, cols))?;
        let (mut out, bytes) = crate::evaluate_integer_with_ax(&ordinary, x, m, read, Some(&q), Some(&ax))?;
        q.append_wire_values(&mut out);
        out.extend_from_slice(&q.scales()[..sx_len]);
        out.extend_from_slice(&ax);
        if !out.iter().all(|v| v.is_finite()) { return Err("projection reuse finite state".into()); }
        Ok((out, bytes + ra))
    } else {
        // Client-provided state is always untrusted: reconstruct the opaque
        // bounded integer type, validate scales and F32 A products every call.
        let q = crate::profile::measure("activation_restore", || int8_kernel::QuantizedRows::from_wire(
            n, cols, &x[..q_len], &x[q_len..q_len + sx_len]))?;
        crate::evaluate_integer_with_ax(&ordinary, &[], m, read, Some(&q), Some(&x[q_len + sx_len..]))
    }
}

pub(crate) fn evaluate_prepared<F,B>(r:&Request,q:&int8_kernel::QuantizedRows,ax:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer,
{
    if !crate::lossless_encoding(&r.encoding) || r.dims.len() != if r.encoding == crate::projection_codec::NAME {5} else {4} || r.aux.len() != 2
        || r.scalars.len() != 1 || !r.scalars[0].is_finite() {
        return Err("projection reuse metadata".into());
    }
    let (n, rows, cols, start) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3]);
    if n == 0 || n > 512 || rows == 0 || rows % 8 != 0 || cols == 0 || cols % 256 != 0
        || rows.checked_mul(cols).is_none_or(|v| v > 30_000_000)
        || n.checked_mul(rows).is_none_or(|v| v > crate::MAX_FLOATS)
        || n.checked_mul(cols).is_none_or(|v| v > crate::MAX_FLOATS) {
        return Err("projection reuse bounds".into());
    }
    let a = m.tensors.iter().find(|t| t.name == r.aux[0]).ok_or("projection reuse A")?;
    let b = m.tensors.iter().find(|t| t.name == r.aux[1]).ok_or("projection reuse B")?;
    let base = m.tensors.iter().find(|t| t.name == r.tensor).ok_or("projection reuse base")?;
    if (r.dims.len()==5 && r.dims[4]!=a.rows) || a.rows == 0 || a.rows > 256 || a.cols != cols || b.cols != a.rows
        || base.dtype != "int8" || base.cols != cols
        || start.checked_add(rows).is_none_or(|v| v > base.rows || v > b.rows)
        || n as u64 * (rows as u64 * cols as u64 + a.rows as u64 * (rows + cols) as u64) > 4_000_000_000 {
        return Err("projection reuse shape/work".into());
    }
    if q.rows()!=n || q.cols()!=cols || ax.len()!=n*a.rows {return Err("prepared projection shape".into());}
    let mut ordinary=r.clone();ordinary.op="lora_integer".into();ordinary.dims.truncate(4);
    crate::evaluate_integer_with_ax(&ordinary,&[],m,read,Some(q),Some(ax))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    #[test]
    fn captured_tiles_match_original_lora_with_unrounded_state() {
        let mut data = vec![];
        let mut tensors = vec![];
        for (name, rows, cols, int8) in [("base",16,256,true),("A",4,256,false),("B",16,4,false)] {
            let offset = data.len() as u64;
            for i in 0..rows*cols {
                if int8 { data.push(((i*17%255) as i16-127) as i8 as u8); }
                else { data.extend_from_slice(&(((i%19) as f32-9.)/37.).to_le_bytes()); }
            }
            if int8 { for i in 0..rows { data.extend_from_slice(&(0.0027 + i as f32*0.0001).to_le_bytes()); } }
            tensors.push(Tensor { name:name.into(), offset, rows, cols, dtype:if int8 {"int8"} else {"f32"}.into(), bytes:data.len() as u64-offset });
        }
        let m = Manifest {version:1, model:"0".repeat(64), pack_hash:"1".repeat(64), bytes:data.len() as u64, tensors};
        for n in [1,7,8,32,132] {
            let x: Vec<_> = (0..n*256).map(|i| crate::bf(((i%29) as f32-14.)/11.)).collect();
            let mut r = Request {version:1, model:m.model.clone(), pack_hash:m.pack_hash.clone(), input_hash:"0".repeat(64), step:0,
                op:"lora_integer".into(), tensor:"base".into(), dims:vec![n,8,256,0], aux:vec!["A".into(),"B".into()], scalars:vec![2.], encoding:"bf16-block256-exact-v1".into()};
            let mut read = |offset:u64, len:usize| Ok(data[offset as usize..offset as usize+len].to_vec());
            let first = crate::evaluate_integer_with_ax(&r,&x,&m,&mut read,None,None).unwrap().0;
            r.op = "lora_integer_capture".into();
            let captured = evaluate(&r,&x,&m,&mut read).unwrap().0;
            assert_eq!(first.iter().map(|x|x.to_bits()).collect::<Vec<_>>(), captured[..n*8].iter().map(|x|x.to_bits()).collect::<Vec<_>>());
            // The lossless codec must preserve F32 scales and A products.
            let blob = crate::encode(&r,&captured[n*8..]).unwrap();
            let (_,state) = crate::decode(&blob).unwrap();
            r.op = "lora_integer".into(); r.dims[3] = 8;
            let second = crate::evaluate_integer_with_ax(&r,&x,&m,&mut read,None,None).unwrap().0;
            r.op = "lora_integer_reuse".into();
            let reused = evaluate(&r,&state,&m,&mut read).unwrap().0;
            assert_eq!(second.iter().map(|x|x.to_bits()).collect::<Vec<_>>(), reused.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
            let mut typed=r.clone();typed.encoding=crate::projection_codec::NAME.into();typed.dims.push(4);
            let packet=crate::encode(&typed,&state).unwrap();
            let (decoded,input)=crate::decode_query(&packet).unwrap();
            assert!(input.values().is_none());
            let direct=crate::evaluate_decoded_with_prepared_buffer(&decoded,&input,&m,&mut read).unwrap().0;
            assert_eq!(second.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),direct.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
            for field in 0..5 {
                let mut different=decoded.clone();
                match field {0=>different.tensor="other".into(),1=>different.step+=1,2=>different.dims[3]=0,3=>different.scalars[0]=3.,_=>different.input_hash="f".repeat(64)}
                assert!(crate::evaluate_decoded_with_prepared_buffer(&different,&input,&m,|_,_|->crate::Result<Vec<u8>> {panic!("mismatched prepared identity read weights")}).is_err());
            }
            let (header,_)=crate::decode_envelope(&packet).unwrap();
            let payload_start=4+u32::from_le_bytes(packet[..4].try_into().unwrap()) as usize;
            for (offset,bytes) in [(1,vec![128]),(1+n*256,vec![0;4]),(1+n*256+n*4,f32::NAN.to_le_bytes().to_vec())] {
                let mut payload=packet[payload_start..packet.len()-32].to_vec();payload[offset..offset+bytes.len()].copy_from_slice(&bytes);
                assert!(crate::projection_codec::PreparedProjection::decode(&header,&payload).is_err());
            }
            for bad in [-128.,128.,0.5,f32::NAN] {
                let mut changed=state.clone(); changed[0]=bad;
                assert!(evaluate(&r,&changed,&m,&mut read).is_err());
            }
            let mut changed=state.clone();changed[n*256]=0.;
            assert!(evaluate(&r,&changed,&m,&mut read).is_err());
            assert!(evaluate(&r,&state[..state.len()-1],&m,&mut read).is_err());
            r.pack_hash="different".into();
            assert!(crate::evaluate_with_reader(&r,&state,&m,&mut read).is_err());
        }
    }
}
