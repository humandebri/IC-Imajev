//! Exact serialization of already quantized integers; F32 scales/A stay raw.
use crate::{Request,Result,MAX_FLOATS};
pub(crate) const NAME: &str = "projection-block256-exact-v1";
/// The frame's bounds, integer lanes and finite A products are checked once.
/// Fields are private, so callers cannot forge or invalidate this type.
/// ```compile_fail
/// let mut state: imajev_runtime::PreparedProjection = todo!();
/// state.ax.clear();
/// ```
pub struct PreparedProjection {
    request: Request,
    q: crate::int8_kernel::QuantizedRows,
    ax: Vec<f32>,
}
impl PreparedProjection {
    pub(crate) fn decode(r:&Request,p:&[u8])->Result<Self> {
        let (_,q,sx,tail)=layout(r)?;
        if !matches!(r.op.as_str(),"lora_integer_reuse"|"mlp_gate_up_reuse") || p.first()!=Some(&2) {return Err("projection codec direction".into());}
        if p.len()!=1+q+tail*4 {return Err("projection codec length".into());}
        let scales:Vec<_>=p[1+q..1+q+sx*4].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        let ax:Vec<_>=p[1+q+sx*4..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        if !ax.iter().all(|v|v.is_finite()) {return Err("invalid activation".into());}
        let input=crate::profile::measure("activation_decode_i8", || crate::int8_kernel::QuantizedRows::from_bytes(r.dims[0],r.dims[2],&p[1..1+q],&scales))?;
        Ok(Self {request:r.clone(),q:input,ax})
    }
    pub(crate) fn evaluate<F,B>(&self,r:&Request,m:&crate::Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:crate::prepared_weights::WeightBuffer,
    {
        let original=&self.request;
        if r.version!=original.version || r.model!=original.model || r.pack_hash!=original.pack_hash
            || r.input_hash!=original.input_hash || r.step!=original.step || r.op!=original.op || r.tensor!=original.tensor
            || r.dims!=original.dims || r.aux!=original.aux || r.encoding!=original.encoding
            || r.scalars.len()!=original.scalars.len() || !r.scalars.iter().zip(&original.scalars).all(|(a,b)|a.to_bits()==b.to_bits()) {
            return Err("prepared projection identity".into());
        }
        if r.op=="mlp_gate_up_reuse" {crate::mlp_reuse::evaluate_prepared(r,&self.q,&self.ax,m,read)}
        else {crate::projection_reuse::evaluate_prepared(r,&self.q,&self.ax,m,read)}
    }
}
fn layout(r:&Request)->Result<(usize,usize,usize,usize)> {
    if r.dims.len()!=5 || !matches!(r.op.as_str(),"lora_integer_capture"|"lora_integer_reuse"|"mlp_gate_up_capture"|"mlp_gate_up_reuse") {
        return Err("projection codec metadata".into());
    }
    let (n,rows,cols,rank)=(r.dims[0],r.dims[1],r.dims[2],r.dims[4]);
    if n==0 || n>512 || rows==0 || rows%8!=0 || cols==0 || cols>262144 || cols%256!=0 || rank==0 || rank>256 {
        return Err("projection codec shape".into());
    }
    let y=n.checked_mul(rows).ok_or("projection codec overflow")?;
    let q=n*cols;let sx=n*(cols/256);let tail=sx+n*rank*if r.op.starts_with("mlp_gate_up_") {2} else {1};
    if y>MAX_FLOATS || q+tail>MAX_FLOATS || (is_capture(r) && y+q+tail>MAX_FLOATS)
        || y==q+tail {return Err("projection codec bounds".into());}
    Ok((y,q,sx,tail))
}
fn is_capture(r:&Request)->bool {matches!(r.op.as_str(),"lora_integer_capture"|"mlp_gate_up_capture")}
pub(crate) fn append(b:&mut Vec<u8>,r:&Request,x:&[f32])->Result<()> {
    let (y,q,sx,tail)=layout(r)?;
    let capture=is_capture(r);
    if (capture && x.len()==q) || (!capture && x.len()==y) {
        b.push(0);return crate::block_codec::append(b,x);
    }
    let prefix=if capture {y} else {0};
    if x.len()!=prefix+q+tail || !crate::bf16_codec::all_bf16(&x[..prefix]) {
        return Err("projection codec values".into());
    }
    if !x[prefix+q..prefix+q+sx].iter().all(|v| v.is_finite() && *v>0.) {return Err("projection codec scales".into());}
    if b.len()+1+prefix*2+q+tail*4+32>2_000_000 {return Err("projection codec size".into());}
    b.push(if capture {1} else {2});
    let begin=b.len();b.resize(begin+prefix*2,0);
    crate::bf16_codec::pack(&x[..prefix],&mut b[begin..]);
    let begin=b.len();b.resize(begin+q,0);
    pack_integers(&x[prefix..prefix+q],&mut b[begin..])?;
    for v in &x[prefix+q..] {b.extend_from_slice(&v.to_le_bytes());}
    Ok(())
}
pub(crate) fn decode(r:&Request,p:&[u8])->Result<Vec<f32>> {
    let (y,q,sx,tail)=layout(r)?;
    let (&tag,body)=p.split_first().ok_or("projection codec tag")?;
    let capture=is_capture(r);
    if tag==0 {
        let x=crate::block_codec::decode(body)?;
        if x.len()!=if capture {q} else {y} {return Err("projection codec plain length".into());}
        return Ok(x);
    }
    if tag!=if capture {1} else {2} {return Err("projection codec direction".into());}
    let prefix=if capture {y} else {0};
    if body.len()!=prefix*2+q+tail*4 {return Err("projection codec length".into());}
    let mut x=vec![0.;prefix+q+tail];
    crate::bf16_codec::unpack(&body[..prefix*2],&mut x[..prefix]);
    unpack_integers(&body[prefix*2..prefix*2+q],&mut x[prefix..prefix+q])?;
    for (dst,bytes) in x[prefix+q..].iter_mut().zip(body[prefix*2+q..].chunks_exact(4)) {
        *dst=f32::from_le_bytes(bytes.try_into().unwrap());
    }
    if !x[prefix+q..prefix+q+sx].iter().all(|v| v.is_finite() && *v>0.) {return Err("projection codec scales".into());}
    Ok(x)
}
pub(crate) fn pack_integers(x:&[f32],out:&mut[u8])->Result<()> {
    #[cfg(target_arch="wasm32")]
    unsafe {return pack_simd(x,out);}
    #[cfg(not(target_arch="wasm32"))]
    {
        for (&v,dst) in x.iter().zip(out) {
            if !v.is_finite() || !(-127. ..=127.).contains(&v) || v!=v.trunc() || v.to_bits()==0x80000000 {return Err("projection codec integer".into());}
            *dst=v as i8 as u8;
        }
        Ok(())
    }
}
fn unpack_integers(x:&[u8],out:&mut[f32])->Result<()> {
    #[cfg(target_arch="wasm32")]
    unsafe {return unpack_simd(x,out);}
    #[cfg(not(target_arch="wasm32"))]
    {
        for (&v,dst) in x.iter().zip(out) {
            if v==128 {return Err("projection codec integer".into());}
            *dst=(v as i8) as f32;
        }
        Ok(())
    }
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn pack_simd(x:&[f32],out:&mut[u8])->Result<()> {
    use core::arch::wasm32::*;
    // layout proves q is a multiple of256, hence every8-lane load/store fits.
    for i in (0..x.len()).step_by(8) {
        let lo=v128_load(x.as_ptr().add(i).cast());let hi=v128_load(x.as_ptr().add(i+4).cast());
        let q0=i32x4_trunc_sat_f32x4(lo);let q1=i32x4_trunc_sat_f32x4(hi);
        for (v,q) in [(lo,q0),(hi,q1)] {
            let exact=f32x4_eq(v,f32x4_convert_i32x4(q));
            let bounds=v128_and(i32x4_ge(q,i32x4_splat(-127)),i32x4_le(q,i32x4_splat(127)));
            let positive_zero=i32x4_ne(v,i32x4_splat(i32::MIN));
            if !i32x4_all_true(v128_and(v128_and(exact,bounds),positive_zero)) {return Err("projection codec integer".into());}
        }
        let q=i8x16_narrow_i16x8(i16x8_narrow_i32x4(q0,q1),i16x8_splat(0));
        out[i..i+8].copy_from_slice(&i64x2_extract_lane::<0>(q).to_le_bytes());
    }
    Ok(())
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn unpack_simd(x:&[u8],out:&mut[f32])->Result<()> {
    use core::arch::wasm32::*;
    for i in (0..x.len()).step_by(16) {
        let q=v128_load(x.as_ptr().add(i).cast());
        if i8x16_bitmask(i8x16_eq(q,i8x16_splat(-128)))!=0 {return Err("projection codec integer".into());}
        let lo=i16x8_extend_low_i8x16(q);let hi=i16x8_extend_high_i8x16(q);
        for (j,v) in [i32x4_extend_low_i16x8(lo),i32x4_extend_high_i16x8(lo),i32x4_extend_low_i16x8(hi),i32x4_extend_high_i16x8(hi)].into_iter().enumerate() {
            v128_store(out.as_mut_ptr().add(i+j*4).cast(),f32x4_convert_i32x4(v));
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request()->Request {Request {version:1,model:"0".repeat(64),pack_hash:"1".repeat(64),input_hash:"0".repeat(64),step:0,op:"lora_integer_capture".into(),tensor:"base".into(),dims:vec![7,8,256,0,4],aux:vec!["A".into(),"B".into()],scalars:vec![2.],encoding:NAME.into()}}
    #[test]
    fn lossless_state_and_plain_frames_are_canonical_and_bounded() {
        let mut r=request();let (y,q,sx,tail)=layout(&r).unwrap();
        let mut x=vec![crate::bf(-3.14);y];x.extend((0..q).map(|i|(i%255) as f32-127.));
        x.extend(vec![f32::from_bits(1);sx]);
        x.extend((0..tail-sx).map(|i|if i%2==0 {-0.} else {f32::from_bits(0x3eaaaaab)}));
        for capture in [true,false] {
            r.op=if capture {"lora_integer_capture"} else {"lora_integer_reuse"}.into();
            let values=if capture {&x[..]} else {&x[y..]};
            let blob=crate::encode(&r,values).unwrap();let (_,got)=crate::decode(&blob).unwrap();
            assert_eq!(values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),got.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let mut payload=vec![];append(&mut payload,&r,values).unwrap();
            let pos=1+if capture {y*2} else {0};payload[pos]=128;
            assert!(decode(&r,&payload).is_err());
            let mut payload=vec![];append(&mut payload,&r,values).unwrap();
            payload[0]=3;assert!(decode(&r,&payload).is_err());
            assert!(decode(&r,&payload[..payload.len()-1]).is_err());
        }
        r.op="lora_integer_capture".into();
        for bad in [-128.,128.,0.5,-0.,f32::NAN] {
            let mut changed=x.clone();changed[y]=bad;assert!(crate::encode(&r,&changed).is_err());
        }
        let plain=vec![f32::from_bits(0x3eaaaaab);q];
        let (_,got)=crate::decode(&crate::encode(&r,&plain).unwrap()).unwrap();assert_eq!(plain,got);
        r.dims[4]=0;assert!(crate::encode(&r,&plain).is_err());
        r.dims=vec![512,262144,262144,0,256];assert!(crate::encode(&r,&[]).is_err());
    }
}
