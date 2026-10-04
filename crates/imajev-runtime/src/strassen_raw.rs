//! Original-size fixed INT8 quadrants. Query-local operand expansion is shared.
use super::PackedView;
use crate::{Result,int8_kernel::QuantizedRows};
#[derive(Debug)]pub(crate) struct Operands{data:Vec<i16>,pairs:usize,cols:usize}
impl Operands{
 pub(crate) fn new(q:&QuantizedRows)->Self{
  let pairs=q.rows().div_ceil(2);let cols=q.cols();let len=7*pairs*cols;
  let mut data=Vec::<i16>::with_capacity(len);
  // QuantizedRows is immutable and checked: cols is divisible by256,
  // rows<=512, cols<=262144, and the input is padded to eight tokens.
  // Each (m,pair,block) owns a disjoint256-element destination span.
  // prepare writes32 complete8-lane vectors at offsets k*2 (k=0,4,..124).
  // No destination load occurs; an odd final pair reads initialized padding.
  // Thus every spare element is written exactly once before committing len.
  #[cfg(target_arch="wasm32")]
  unsafe{prepare(q,pairs,data.as_mut_ptr());}
  #[cfg(not(target_arch="wasm32"))]
  for pair in 0..pairs{for block in 0..cols/256{for k in 0..128{
   let p=pair*2*cols+block*256+k;let(a,b,c,d)=(q.values()[p],q.values()[p+128],q.values()[p+cols],q.values()[p+cols+128]);
   for(m,v)in[a+d,c+d,a,d,a+b,c-a,b-d].into_iter().enumerate(){let i=m*pairs*cols+pair*cols+block*256+(k/4)*8+k%4;data.spare_capacity_mut()[i].write(v);data.spare_capacity_mut()[i+4].write(v);}
  }}}
  // SAFETY: both target paths above initialize every element in0..len.
  // The Vec remains len0 throughout the writes, including on unwinding.
  unsafe { data.set_len(len); }
  Self{data,pairs,cols}
 }
}
// The existing nine-argument WAT ABI uses the full column stride. Keep that
// allocation, but only transform blocks this continuation actually consumes.
// Only selected blocks are initialized/read; unused slots are MaybeUninit.
#[cfg(feature="experimental-int8-k-continue")]
struct RangeOperands { data:Vec<core::mem::MaybeUninit<i16>>,pairs:usize,cols:usize }
#[cfg(feature="experimental-int8-k-continue")]
impl RangeOperands {
 fn new(q:&QuantizedRows,begin:usize,count:usize)->Result<Self> {
  let cols=q.cols();
  if begin%256!=0 || count==0 || count%256!=0 || begin.checked_add(count).is_none_or(|end|end>cols) {
   return Err("S1 operand range".into());
  }
  let pairs=q.rows().div_ceil(2);
  let mut data=vec![core::mem::MaybeUninit::<i16>::uninit();7*pairs*cols];
  #[cfg(target_arch="wasm32")]
  // SAFETY: q is padded to eight initialized tokens. Range is within its
  // checked block256 stride; each vector store stays in a disjoint block.
  // Only this selected range is subsequently loaded, with unchanged block bounds.
  unsafe { prepare_range(q,pairs,data.as_mut_ptr().cast::<i16>(),begin/256,(begin+count)/256); }
  #[cfg(not(target_arch="wasm32"))]
  for pair in 0..pairs { for block in begin/256..(begin+count)/256 { for k in 0..128 {
   let p=pair*2*cols+block*256+k;
   let(a,b,c,d)=(q.values()[p],q.values()[p+128],q.values()[p+cols],q.values()[p+cols+128]);
   for(m,v)in[a+d,c+d,a,d,a+b,c-a,b-d].into_iter().enumerate(){
    let i=m*pairs*cols+pair*cols+block*256+(k/4)*8+k%4;
    data[i].write(v);data[i+4].write(v);
   }
  }}}
  Ok(Self{data,pairs,cols})
 }
}
#[cfg(all(target_arch="wasm32",feature="experimental-int8-k-continue"))]
#[target_feature(enable="simd128")]
unsafe fn prepare_range(q:&QuantizedRows,pairs:usize,out:*mut i16,first:usize,end:usize) {
 use core::arch::wasm32::*;
 let cols=q.cols();
 for pair in 0..pairs { for block in first..end {
  let p=q.values().as_ptr().add(pair*2*cols+block*256);
  let dst:[*mut i16;7]=core::array::from_fn(|m|out.add(m*pairs*cols+pair*cols+block*256));
  for k in (0..128).step_by(4) {
   let a=v128_load64_splat(p.add(k).cast());let b=v128_load64_splat(p.add(128+k).cast());
   let c=v128_load64_splat(p.add(cols+k).cast());let d=v128_load64_splat(p.add(cols+128+k).cast());
   for(m,v)in[i16x8_add(a,d),i16x8_add(c,d),a,d,i16x8_add(a,b),i16x8_sub(c,a),i16x8_sub(b,d)].into_iter().enumerate(){v128_store(dst[m].add(k*2).cast(),v);}
  }
 }}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn prepare(q:&QuantizedRows,pairs:usize,out:*mut i16){use core::arch::wasm32::*;let cols=q.cols();for pair in 0..pairs{for block in 0..cols/256{
 let p=q.values().as_ptr().add(pair*2*cols+block*256);let dst:[*mut i16;7]=core::array::from_fn(|m|out.add(m*pairs*cols+pair*cols+block*256));
 for k in(0..128).step_by(4){let a=v128_load64_splat(p.add(k).cast());let b=v128_load64_splat(p.add(128+k).cast());let c=v128_load64_splat(p.add(cols+k).cast());let d=v128_load64_splat(p.add(cols+128+k).cast());for(m,v)in[i16x8_add(a,d),i16x8_add(c,d),a,d,i16x8_add(a,b),i16x8_sub(c,a),i16x8_sub(b,d)].into_iter().enumerate(){v128_store(dst[m].add(k*2).cast(),v);}}
 }}}
pub(crate) fn pack(bytes:&[u8],rows:usize,cols:usize)->Vec<u8>{let stride=rows/4*cols;let mut data=vec![0;rows*cols];for quartet in 0..rows/4{for pair in 0..2{for block in 0..cols/256{for k in(0..128).step_by(4){let p=(quartet*4+pair*2)*cols+block*256+k;let dst=quartet*cols+block*256+k*2+pair*4;for(m,start)in[p,p+128,p+cols,p+cols+128].into_iter().enumerate(){data[m*stride+dst..m*stride+dst+4].copy_from_slice(&bytes[start..start+4]);}}}}}data.extend_from_slice(&bytes[rows*cols..]);data}
pub(crate) fn index(rows:usize,cols:usize,row:usize,c:usize)->usize{let m=(row%2)*2+(c%256)/128;let k=c%128;m*(rows/4)*cols+(row/4)*cols+(c/256)*256+(k/4)*8+((row%4)/2)*4+k%4}
pub(crate) fn project(q:&QuantizedRows,w:&PackedView,sw:&[f32])->Result<Vec<f32>>{
 let a=q.strassen_operands();debug_assert_eq!(a.cols,q.cols());let cols=q.cols();let rows=w.rows();let start=w.start/cols;let fixed=&w.fixed;let stride=fixed.rows/4*cols;let mut out=vec![0.;q.rows()*rows];
 #[cfg(target_arch="wasm32")]
 {let ap:[*const i16;7]=core::array::from_fn(|m|a.data.as_ptr().wrapping_add(m*a.pairs*cols));let tile=if cfg!(feature="experimental-strassen-output128") && rows%128==0 && start%4==0 {128}else{32};for r in(0..rows).step_by(tile){let width=(rows-r).min(tile);let pad;let scales;let wp:[*const u8;4];let s:&[f32];
  if width==tile && (start+r)%4==0 {wp=core::array::from_fn(|m|fixed.data.as_ptr().wrapping_add(m*stride+(start+r)/4*cols));s=&sw[r..r+tile];}
  else{pad={let mut b=vec![0;tile*cols];for m in 0..4{if (start+r)%4==0 {let i=m*stride+(start+r)/4*cols;let len=width/4*cols;b[m*(tile/4)*cols..m*(tile/4)*cols+len].copy_from_slice(&fixed.data[i..i+len]);}
   else {for row in 0..width{for c in 0..cols{if (row%2)*2+(c%256)/128==m{let dst=index(tile,cols,row,c);b[dst]=fixed.data[index(fixed.rows,cols,start+r+row,c)];}}}}}b};wp=core::array::from_fn(|m|pad.as_ptr().wrapping_add(m*(tile/4)*cols));scales={let mut b=vec![1.;tile];b[..width].copy_from_slice(&sw[r..]);b};s=&scales;}
  let mut sums=vec![0f32;q.rows()*tile];for block in 0..cols/256{unsafe{
   #[cfg(feature="experimental-strassen-output128")]
   if tile==128 {accumulate_wide(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,s.as_ptr(),sums.as_mut_ptr(),q.rows());continue;}
   accumulate(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,s.as_ptr(),sums.as_mut_ptr(),q.rows());
  }}
  for t in 0..q.rows(){out[t*rows+r..t*rows+r+width].copy_from_slice(&sums[t*tile..t*tile+width]);}
 }}
 #[cfg(not(target_arch="wasm32"))]
 {for pair in 0..a.pairs{for row in 0..rows/2{for block in 0..cols/256{let mut p=[0i32;7];for k in 0..128{let global=start+row*2;let i=(global/4)*cols+block*256+(k/4)*8+((global%4)/2)*4+k%4;let(b,c,d,e)=(fixed.data[i]as i8 as i16,fixed.data[stride+i]as i8 as i16,fixed.data[2*stride+i]as i8 as i16,fixed.data[3*stride+i]as i8 as i16);for(m,v)in[b+e,b,d-e,c-b,e,b+d,c+e].into_iter().enumerate(){p[m]+=a.data[m*a.pairs*cols+pair*cols+block*256+(k/4)*8+k%4]as i32*v as i32;}}
  let dots=[[p[0]+p[3]-p[4]+p[6],p[2]+p[4]],[p[1]+p[3],p[0]-p[1]+p[2]+p[5]]];for ti in 0..2{let t=pair*2+ti;if t<q.rows(){for ri in 0..2{let r=row*2+ri;out[t*rows+r]+=(dots[ti][ri]as f32*q.scales()[t*(cols/256)+block])*sw[r];}}}
 }}}}
 if out.iter().any(|v|!v.is_finite()){return Err("raw S1 projection finite".into());}Ok(out)
}
#[cfg(target_arch="wasm32")]
#[export_name="__imajev_s1_raw_accumulate"]#[inline(never)]
unsafe extern "C" fn accumulate(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;for i in 0..n*32{core::ptr::write_volatile(sums.add(i),f32::from_bits(marker|0x7fc00000));}}

// The128-output kernel has the same K256 integer reductions, scale multiplies,
// and block summation order. Only fully initialized, aligned128-row views use
// this ABI; all partial/small views retain the32-output implementation.
#[cfg(all(target_arch="wasm32",feature="experimental-strassen-output128"))]
#[export_name="__imajev_s1_wide_accumulate"]#[inline(never)]
unsafe extern "C" fn accumulate_wide(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:*mut f32,n:usize){let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(sums as usize)^n)as u32;for i in 0..n*128{core::ptr::write_volatile(sums.add(i),f32::from_bits(marker|0x7fc00000));}}

#[cfg(test)]
mod tests {
 #[cfg(feature="experimental-int8-k-continue")]
 #[test]
 fn ranged_operands_match_full_selected_blocks_without_reading_inactive() {
  for cols in [512,4096] { for n in [1,7,87,89] {
   let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,0.,-0.,1.,-1.][i%6]).collect();
   let q=crate::int8_kernel::quantize_rows(&x,n,cols).unwrap();
   let full=q.strassen_operands();
   for (begin,count) in [(0,256),(256,cols-256),(cols-256,256),(0,cols)] {
    let part=super::RangeOperands::new(&q,begin,count).unwrap();
    for m in 0..7 { for pair in 0..part.pairs { for block in 0..cols/256 {
     let i=m*part.pairs*cols+pair*cols+block*256;
     if (begin/256..(begin+count)/256).contains(&block) {for k in 0..256 {assert_eq!(unsafe {part.data[i+k].assume_init()},full.data[i+k]);}}
     // Inactive slots are deliberately not loaded.
    }}}
   }
   for (b,c) in [(1,256),(0,0),(cols,256),(usize::MAX-255,256)] {assert!(super::RangeOperands::new(&q,b,c).is_err());}
  }}
 }
 #[test]
 fn complete_operand_writes_cover_odd_tail_and_reuse() {
  for cols in [256,512] {for n in [1,7,8,45,87,132,512] {
   let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,0.,-0.,1.,-1.][i%6]).collect();
   let q=crate::int8_kernel::quantize_rows(&x,n,cols).unwrap();
   let a=q.strassen_operands();let mut seen=vec![0u8;a.data.len()];
   for m in 0..7 {for pair in 0..a.pairs {for block in 0..cols/256 {for k in 0..128 {
    let p=pair*2*cols+block*256+k;
    let (v0,v1,v2,v3)=(q.values()[p],q.values()[p+128],q.values()[p+cols],q.values()[p+cols+128]);
    let expected=[v0+v3,v2+v3,v0,v3,v0+v1,v2-v0,v1-v3][m];
    let i=m*a.pairs*cols+pair*cols+block*256+(k/4)*8+k%4;
    for j in [i,i+4] {seen[j]+=1;assert_eq!(a.data[j],expected);}
   }}}}
   assert!(seen.iter().all(|v|*v==1));
   assert!(core::ptr::eq(a,q.strassen_operands()));
  }}
 }
}

// Keep the adopted full projection as a same-module comparison. This path
// carries its original F32 block sum and never evaluates previous blocks.
#[cfg(feature="experimental-int8-k-continue")]
pub(crate) fn project_continued(q:&QuantizedRows,w:&PackedView,sw:&[f32],begin:usize,count:usize,initial:&[f32])->Result<Vec<f32>>{
 let a=crate::profile::measure("integer_k_prepare_new_columns",||RangeOperands::new(q,begin,count))?;debug_assert_eq!(a.cols,q.cols());let cols=q.cols();let rows=w.rows();let start=w.start/cols;let fixed=&w.fixed;let stride=fixed.rows/4*cols;let mut out=initial.to_vec();
 #[cfg(target_arch="wasm32")]
 {let ap:[*const i16;7]=core::array::from_fn(|m|a.data.as_ptr().wrapping_add(m*a.pairs*cols).cast::<i16>());let tile=if cfg!(feature="experimental-strassen-output128") && rows%128==0 && start%4==0 {128}else{32};for r in(0..rows).step_by(tile){let width=(rows-r).min(tile);let pad;let scales;let wp:[*const u8;4];let s:&[f32];
  if width==tile && (start+r)%4==0 {wp=core::array::from_fn(|m|fixed.data.as_ptr().wrapping_add(m*stride+(start+r)/4*cols));s=&sw[r..r+tile];}
  else{pad={let mut b=vec![0;tile*cols];for m in 0..4{if (start+r)%4==0 {let i=m*stride+(start+r)/4*cols;let len=width/4*cols;b[m*(tile/4)*cols..m*(tile/4)*cols+len].copy_from_slice(&fixed.data[i..i+len]);}
   else {for row in 0..width{for c in 0..cols{if (row%2)*2+(c%256)/128==m{let dst=index(tile,cols,row,c);b[dst]=fixed.data[index(fixed.rows,cols,start+r+row,c)];}}}}}b};wp=core::array::from_fn(|m|pad.as_ptr().wrapping_add(m*(tile/4)*cols));scales={let mut b=vec![1.;tile];b[..width].copy_from_slice(&sw[r..]);b};s=&scales;}
  let mut sums=vec![0f32;q.rows()*tile];for t in 0..q.rows(){sums[t*tile..t*tile+width].copy_from_slice(&initial[t*rows+r..t*rows+r+width]);}for block in begin/256..(begin+count)/256{unsafe{
   #[cfg(feature="experimental-strassen-output128")]
   if tile==128 {accumulate_wide(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,s.as_ptr(),sums.as_mut_ptr(),q.rows());continue;}
   accumulate(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,s.as_ptr(),sums.as_mut_ptr(),q.rows());
  }}
  for t in 0..q.rows(){out[t*rows+r..t*rows+r+width].copy_from_slice(&sums[t*tile..t*tile+width]);}
 }}
 #[cfg(not(target_arch="wasm32"))]
 {for pair in 0..a.pairs{for row in 0..rows/2{for block in begin/256..(begin+count)/256{let mut p=[0i32;7];for k in 0..128{let global=start+row*2;let i=(global/4)*cols+block*256+(k/4)*8+((global%4)/2)*4+k%4;let(b,c,d,e)=(fixed.data[i]as i8 as i16,fixed.data[stride+i]as i8 as i16,fixed.data[2*stride+i]as i8 as i16,fixed.data[3*stride+i]as i8 as i16);for(m,v)in[b+e,b,d-e,c-b,e,b+d,c+e].into_iter().enumerate(){p[m]+=unsafe {a.data[m*a.pairs*cols+pair*cols+block*256+(k/4)*8+k%4].assume_init()}as i32*v as i32;}}
  let dots=[[p[0]+p[3]-p[4]+p[6],p[2]+p[4]],[p[1]+p[3],p[0]-p[1]+p[2]+p[5]]];for ti in 0..2{let t=pair*2+ti;if t<q.rows(){for ri in 0..2{let r=row*2+ri;out[t*rows+r]+=(dots[ti][ri]as f32*q.scales()[t*(cols/256)+block])*sw[r];}}}
 }}}}
 if out.iter().any(|v|!v.is_finite()){return Err("raw S1 projection finite".into());}Ok(out)
}
