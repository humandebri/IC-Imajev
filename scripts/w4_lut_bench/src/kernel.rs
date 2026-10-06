//! Experimental W4A8, exact integer LUT4 sums, no LUT quantization.
//! Layout: output16/K4/four weight bits packed into two 16-byte planes.
//! Each table entry is a signed I16 sum of four activations (-508..508).
pub fn quantize(x: &[f32], cols: usize) -> (Vec<i16>, Vec<f32>) {
    assert!(cols > 0 && cols % 256 == 0 && x.len() % cols == 0);
    let mut q = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len()/256);
    for block in x.chunks_exact(256) {
        assert!(block.iter().all(|v| v.is_finite()));
        let peak=block.iter().fold(0f32, |a,v| a.max(v.abs()));
        let scale=if peak==0. {1.} else {(peak/127.).max(f32::from_bits(1))};
        scales.push(scale);
        q.extend(block.iter().map(|v|(v/scale).round_ties_even().clamp(-127.,127.) as i16));
    }
    (q,scales)
}
pub fn tables(q: &[i16]) -> Vec<u8> {
    assert!(q.len()%4==0 && q.iter().all(|v|(-127..=127).contains(v)));
    let mut out=vec![0;q.len()/4*32];
    for (g,x) in q.chunks_exact(4).enumerate() {
        let mut sums=[0i16;16];
        for mask in 1usize..16 {
            let bit=mask.trailing_zeros() as usize;
            sums[mask]=sums[mask & (mask-1)]+x[bit];
        }
        for (mask,v) in sums.iter().enumerate() {
            let b=v.to_le_bytes();out[g*32+mask]=b[0];out[g*32+16+mask]=b[1];
        }
    }
    out
}
pub fn pack(w: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert!(rows>0 && rows%16==0 && cols%256==0 && w.len()==rows*cols);
    assert!(w.iter().all(|v|(-8..=7).contains(v)));
    let mut out=vec![0;w.len()/2];
    for r in (0..rows).step_by(16) {for k in (0..cols).step_by(4) {
        let off=(r/16*(cols/4)+k/4)*32;
        for lane in 0..16 {let mut indices=[0u8;4];
            for j in 0..4 {let v=w[(r+lane)*cols+k+j] as u8;
                for bit in 0..4 {indices[bit]|=((v>>bit)&1)<<j;}
            }
            out[off+lane]=indices[0]|(indices[1]<<4);
            out[off+16+lane]=indices[2]|(indices[3]<<4);
        }
    }}out
}
pub fn dense(q: &[i16], sx:&[f32], w:&[i8], sw:&[f32], n:usize, rows:usize, cols:usize)->Vec<f32>{
    assert_eq!(q.len(),n*cols);assert_eq!(w.len(),rows*cols);
    assert_eq!(sx.len(),n*(cols/256));assert_eq!(sw.len(),rows*(cols/256));
    #[cfg(target_arch="wasm32")]
    {return unsafe {dense_simd(q,sx,w,sw,n,rows,cols)};}
    #[cfg(not(target_arch="wasm32"))]
    {
    let blocks=cols/256;let mut out=vec![0.;n*rows];
    for t in 0..n {for r in 0..rows {let mut sum=0.;for b in 0..blocks {
        let mut dot=0i32;
        for k in b*256..(b+1)*256 {dot+=q[t*cols+k] as i32*w[r*cols+k] as i32;}
        sum+=(dot as f32*sx[t*blocks+b])*sw[r*blocks+b];
    }out[t*rows+r]=sum;}}out
    }
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn dense_simd(q:&[i16],sx:&[f32],w:&[i8],sw:&[f32],n:usize,rows:usize,cols:usize)->Vec<f32>{
    use core::arch::wasm32::*;
    let mut out=vec![0.;n*rows];let blocks=cols/256;
    for t in 0..n{for r in 0..rows{let mut sum=0.;for b in 0..blocks{
        let mut acc=i32x4_splat(0);
        for k in (b*256..(b+1)*256).step_by(8){
            let x=v128_load(q.as_ptr().add(t*cols+k).cast());
            let weight=i16x8_extend_low_i8x16(v128_load64_zero(w.as_ptr().add(r*cols+k).cast()));
            acc=i32x4_add(acc,i32x4_dot_i16x8(x,weight));
        }
        let dot=i32x4_extract_lane::<0>(acc)+i32x4_extract_lane::<1>(acc)+i32x4_extract_lane::<2>(acc)+i32x4_extract_lane::<3>(acc);
        sum+=(dot as f32*sx[t*blocks+b])*sw[r*blocks+b];
    }out[t*rows+r]=sum;}}out
}
#[cfg(not(target_arch="wasm32"))]
pub fn lut(q:&[i16], sx:&[f32], packed:&[u8], sw:&[f32], n:usize,rows:usize,cols:usize,tab:&[u8])->Vec<f32>{
    assert_eq!(q.len(),n*cols);let mut out=vec![0.;n*rows];let blocks=cols/256;
    for t in 0..n {for r in 0..rows {let mut sum=0.;for b in 0..blocks {let mut dot=0i32;
        for k in (b*256..(b+1)*256).step_by(4) {let off=(r/16*(cols/4)+k/4)*32+r%16;
            let p=packed[off];let z=packed[off+16];let start=(t*cols/4+k/4)*32;
            let read=|i:usize|i16::from_le_bytes([tab[start+i],tab[start+16+i]]) as i32;
            dot+=read((p&15)as usize)+2*read((p>>4)as usize)+4*read((z&15)as usize)-8*read((z>>4)as usize);
        }sum+=(dot as f32*sx[t*blocks+b])*sw[r*blocks+b];
    }out[t*rows+r]=sum;}}out
}
#[cfg(target_arch="wasm32")]
pub fn lut(q:&[i16],sx:&[f32],packed:&[u8],sw:&[f32],n:usize,rows:usize,cols:usize,tab:&[u8])->Vec<f32>{
    assert_eq!(q.len(),n*cols);assert_eq!(packed.len(),rows*cols/2);assert_eq!(tab.len(),n*cols*8);
    assert_eq!(sw.len(),rows*(cols/256));assert_eq!(sx.len(),n*(cols/256));assert!(rows%16==0);
    // SAFETY: dimensions above prove all table/weight loads and output stores
    // fit their buffers. Integer partial sums fit I32; group sums fit I16.
    unsafe {lut_simd(sx,packed,sw,n,rows,cols,tab)}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn lut_simd(sx:&[f32],packed:&[u8],sw:&[f32],n:usize,rows:usize,cols:usize,tab:&[u8])->Vec<f32>{
    use core::arch::wasm32::*;
    #[inline(always)] unsafe fn read(lo:v128,hi:v128,index:v128)->(v128,v128){
        let l=i8x16_swizzle(lo,index);let h=i8x16_swizzle(hi,index);
        // Interleave low/high bytes directly: no extend/shift/or reconstruction.
        (i8x16_shuffle::<0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23>(l,h),
         i8x16_shuffle::<8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31>(l,h))
    }
    let mut out=vec![0.;n*rows];let blocks=cols/256;let mask=i8x16_splat(15);
    for t in 0..n {for r in (0..rows).step_by(16) {for b in 0..blocks {
        let mut a=i32x4_splat(0);let mut c=a;let mut d=a;let mut e=a;
        let table=tab.as_ptr().add((t*cols/4+b*64)*32);
        let weights=packed.as_ptr().add((r/16*(cols/4)+b*64)*32);
        macro_rules! step {($off:expr)=>{{
            let lo=v128_load(table.add($off).cast());let hi=v128_load(table.add($off+16).cast());
            let p=v128_load(weights.add($off).cast());let z=v128_load(weights.add($off+16).cast());
            let (v0,v1)=read(lo,hi,v128_and(p,mask));let (w0,w1)=read(lo,hi,u8x16_shr(p,4));
            let (x0,x1)=read(lo,hi,v128_and(z,mask));let (y0,y1)=read(lo,hi,u8x16_shr(z,4));
            let l=i16x8_sub(i16x8_add(i16x8_add(v0,i16x8_shl(w0,1)),i16x8_shl(x0,2)),i16x8_shl(y0,3));
            let h=i16x8_sub(i16x8_add(i16x8_add(v1,i16x8_shl(w1,1)),i16x8_shl(x1,2)),i16x8_shl(y1,3));
            a=i32x4_add(a,i32x4_extend_low_i16x8(l));c=i32x4_add(c,i32x4_extend_high_i16x8(l));
            d=i32x4_add(d,i32x4_extend_low_i16x8(h));e=i32x4_add(e,i32x4_extend_high_i16x8(h));
        }}}
        // Constant offsets remove per-K4 index checks and loop bookkeeping.
        step!(0);step!(32);step!(64);step!(96);step!(128);step!(160);step!(192);step!(224);
        step!(256);step!(288);step!(320);step!(352);step!(384);step!(416);step!(448);step!(480);
        step!(512);step!(544);step!(576);step!(608);step!(640);step!(672);step!(704);step!(736);
        step!(768);step!(800);step!(832);step!(864);step!(896);step!(928);step!(960);step!(992);
        step!(1024);step!(1056);step!(1088);step!(1120);step!(1152);step!(1184);step!(1216);step!(1248);
        step!(1280);step!(1312);step!(1344);step!(1376);step!(1408);step!(1440);step!(1472);step!(1504);
        step!(1536);step!(1568);step!(1600);step!(1632);step!(1664);step!(1696);step!(1728);step!(1760);
        step!(1792);step!(1824);step!(1856);step!(1888);step!(1920);step!(1952);step!(1984);step!(2016);
        let mut dots=[0i32;16];v128_store(dots.as_mut_ptr().cast(),a);v128_store(dots.as_mut_ptr().add(4).cast(),c);
        v128_store(dots.as_mut_ptr().add(8).cast(),d);v128_store(dots.as_mut_ptr().add(12).cast(),e);
        for lane in 0..16 {out[t*rows+r+lane]+=(dots[lane] as f32*sx[t*blocks+b])*sw[(r+lane)*blocks+b];}
    }}}out
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn all_signed_weights_and_extreme_activations(){
        let rows=16;let cols=512;
        for n in [1,2,3,7,8,9] {
            let q:Vec<i16>=(0..n*cols).map(|i|[-127,127,0,-1,1][i%5]).collect();
            let w:Vec<i8>=(0..rows*cols).map(|i|(i%16)as i8-8).collect();
            let sx=vec![0.013;n*2];let sw=vec![0.019;rows*2];let tab=tables(&q);
            let a=dense(&q,&sx,&w,&sw,n,rows,cols);let b=lut(&q,&sx,&pack(&w,rows,cols),&sw,n,rows,cols,&tab);
            assert!(a.iter().zip(&b).all(|(a,b)|a.to_bits()==b.to_bits()));
        }
    }
}
