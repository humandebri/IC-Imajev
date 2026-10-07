//! Exact K2/four-column fixed INT8 layout and compact rank7 operands.
use super::PackedView;
use crate::{Result,int8_kernel::QuantizedRows};
#[cfg(target_arch="wasm32")]
#[path="win_kernel.rs"]mod win_kernel;
#[derive(Debug)]pub(crate) struct Operands{data:Vec<i16>,pairs:usize,cols:usize}
impl Operands{
 pub(crate) fn new(q:&QuantizedRows)->Self{
  let pairs=q.rows().div_ceil(2);let cols=q.cols();let len=7*pairs*(cols/2);
  let mut data=Vec::<i16>::with_capacity(len);
  unsafe{prepare(q,pairs,data.as_mut_ptr(),0,cols/256);data.set_len(len);}
  Self{data,pairs,cols}
 }
}
// Each selected block owns128 I16 values; range consumers load only written blocks.
unsafe fn prepare(q:&QuantizedRows,pairs:usize,out:*mut i16,first:usize,end:usize){
 let cols=q.cols();
 for pair in 0..pairs{for block in first..end{
  #[cfg(target_arch="wasm32")]
  {use core::arch::wasm32::*;let p=q.values().as_ptr().add(pair*2*cols+block*256);
   for k in(0..128).step_by(4){
    let a=v128_load64_splat(p.add(k).cast());let b=v128_load64_splat(p.add(128+k).cast());
    let c=v128_load64_splat(p.add(cols+k).cast());let d=v128_load64_splat(p.add(cols+128+k).cast());
    let cd=i16x8_add(c,d);let cda=i16x8_sub(cd,a);
    for(m,v)in[a,b,i16x8_sub(b,cda),d,cd,cda,i16x8_sub(a,c)].into_iter().enumerate(){
     v128_store64_lane::<0>(v,out.add(m*pairs*(cols/2)+pair*(cols/2)+block*128+k).cast());
    }
   }
  }
  #[cfg(not(target_arch="wasm32"))]
  for k in 0..128{
   let p=pair*2*cols+block*256+k;let(a,b,c,d)=(q.values()[p],q.values()[p+128],q.values()[p+cols],q.values()[p+cols+128]);
   for(m,v)in[a,b,a+b-c-d,d,c+d,-a+c+d,a-c].into_iter().enumerate(){
    out.add(m*pairs*(cols/2)+pair*(cols/2)+block*128+k).write(v);
   }
  }
 }}
}
pub(crate) fn index(rows:usize,cols:usize,row:usize,c:usize)->usize{
 let m=(row%2)*2+(c%256)/128;
 m*(rows/4)*cols+(row/8)*2*cols+(c/256)*512+((c%128)/2)*8+((row%8)/2)*2+c%2
}
pub(crate) fn pack(bytes:&[u8],rows:usize,cols:usize)->Vec<u8>{
 let mut data=vec![0;rows*cols];
 let stride=rows/4*cols;
 for group in 0..rows/8{for pair in 0..4{for block in 0..cols/256{for k in(0..128).step_by(2){
  let p=(group*8+pair*2)*cols+block*256+k;let dst=group*2*cols+block*512+(k/2)*8+pair*2;
  for(m,start)in[p,p+128,p+cols,p+cols+128].into_iter().enumerate(){data[m*stride+dst..m*stride+dst+2].copy_from_slice(&bytes[start..start+2]);}
 }}}}
 data.extend_from_slice(&bytes[rows*cols..]);data
}
pub(crate) fn project(q:&QuantizedRows,w:&PackedView,sw:&[f32])->Result<Vec<f32>>{
 if q.rows()==1{return single(q,w,sw,0,q.cols()/256,vec![0.;w.rows()]);}
 let a=q.strassen_operands();debug_assert_eq!(a.cols,q.cols());
 Ok(project_blocks(q,w,sw,a.data.as_ptr(),a.pairs,0,q.cols()/256,None))
}
#[cfg(feature="experimental-int8-k-continue")]
pub(crate) fn project_continued(q:&QuantizedRows,w:&PackedView,sw:&[f32],begin:usize,count:usize,initial:&[f32])->Result<Vec<f32>>{
 let first=begin/256;let end=(begin+count)/256;
 if q.rows()==1{return single(q,w,sw,first,end,initial.to_vec());}
 let pairs=q.rows().div_ceil(2);let len=7*pairs*(q.cols()/2);
 let data=crate::profile::measure("integer_k_prepare_new_columns",||{
  let mut data=Vec::<core::mem::MaybeUninit<i16>>::with_capacity(len);
  unsafe{data.set_len(len);prepare(q,pairs,data.as_mut_ptr().cast(),first,end);}data
 });
 let out=project_blocks(q,w,sw,data.as_ptr().cast(),pairs,first,end,Some(initial));
 if !crate::all_finite(&out){return Err("raw S1 projection finite".into());}Ok(out)
}
fn project_blocks(q:&QuantizedRows,w:&PackedView,sw:&[f32],data:*const i16,pairs:usize,first:usize,end:usize,initial:Option<&[f32]>)->Vec<f32>{
 let cols=q.cols();let rows=w.rows();let start=w.start/cols;let fixed=&w.fixed;
 #[cfg(not(target_arch="wasm32"))]
 {let _=(data,pairs);let mut out=initial.map_or_else(||vec![0.;q.rows()*rows],|v|v.to_vec());
  for t in 0..q.rows(){for r in 0..rows{for b in first..end{
   let mut dot=0i32;for k in 0..256{dot+=q.values()[t*cols+b*256+k]as i32*(fixed.data[index(fixed.rows,cols,start+r,b*256+k)]as i8 as i32);}
   out[t*rows+r]+=(dot as f32*q.scales()[t*(cols/256)+b])*sw[r];
  }}}out
 }
 #[cfg(target_arch="wasm32")]
 unsafe{
  let count=q.rows()*rows;let mut out=Vec::<core::mem::MaybeUninit<f32>>::with_capacity(count);out.set_len(count);
  let output=out.as_mut_ptr().cast::<f32>();
  if let Some(v)=initial{core::ptr::copy_nonoverlapping(v.as_ptr(),output,count);}
  let ap:[*const i16;7]=core::array::from_fn(|m|data.add(m*pairs*(cols/2)));
  let stride=fixed.rows/4*cols;let wide=rows%128==0&&start%8==0;let mut r=0;
  while r<rows{
   let tile=if wide&&matches!(rows,8192|4096)&&rows-r>=168{168}else if wide&&!matches!(rows,8192|4096)&&rows-r>=160{160}else if wide&&rows-r>=128{128}else{32};
   let width=(rows-r).min(tile);let pad;let scales;let wp:[*const u8;4];let s:&[f32];
   if width==tile&&(start+r)%8==0{
    wp=core::array::from_fn(|m|fixed.data.as_ptr().add(m*stride+(start+r)/8*2*cols));s=&sw[r..r+tile];
   }else{
    pad={let mut b=vec![0;tile*cols];for row in 0..width{for c in 0..cols{b[index(tile,cols,row,c)]=fixed.data[index(fixed.rows,cols,start+r+row,c)];}}b};
    wp=core::array::from_fn(|m|pad.as_ptr().add(m*(tile/4)*cols));
    scales={let mut b=vec![1.;tile];b[..width].copy_from_slice(&sw[r..r+width]);b};s=&scales;
   }
   let mut scratch=Vec::<f32>::new();
   let (yp,output_rows)=if width==tile{(output.add(r),rows)}else{
    scratch.resize(q.rows()*tile,0.);
    if let Some(v)=initial{for t in 0..q.rows(){scratch[t*tile..t*tile+width].copy_from_slice(&v[t*rows+r..t*rows+r+width]);}}
    (scratch.as_mut_ptr(),tile)
   };
   for b in first..end{
    let seed=initial.is_none()&&b==first;
    let f=match(tile,seed){(168,true)=>win_kernel::wide168_seed,(168,false)=>win_kernel::wide168,(160,true)=>win_kernel::wide160_seed,(160,false)=>win_kernel::wide160,(128,true)=>win_kernel::wide_seed,(128,false)=>win_kernel::wide,(32,true)=>win_kernel::wide32_seed,_=>win_kernel::wide32};
    f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,b*256,q.scales().as_ptr().add(b),output_rows,s.as_ptr(),yp,q.rows());
   }
   if width!=tile{for t in 0..q.rows(){core::ptr::copy_nonoverlapping(scratch.as_ptr().add(t*tile),output.add(t*rows+r),width);}}
   r+=width;
  }
  // Every token/row was written by a seed or copied from initialized carry.
  let mut out=core::mem::ManuallyDrop::new(out);Vec::from_raw_parts(out.as_mut_ptr().cast::<f32>(),out.len(),out.capacity())
 }
}
fn single(q:&QuantizedRows,w:&PackedView,sw:&[f32],first:usize,end:usize,mut out:Vec<f32>)->Result<Vec<f32>>{
 let cols=q.cols();let rows=w.rows();let start=w.start/cols;let fixed=&w.fixed;
 #[cfg(target_arch="wasm32")]
 if start%8==0{unsafe{single_simd(q,w,sw,&mut out,first,end);}}
 #[cfg(target_arch="wasm32")]let scalar=start%8!=0;
 #[cfg(not(target_arch="wasm32"))]let scalar=true;
 if scalar{for r in 0..rows{for b in first..end{let mut dot=0i32;for k in 0..256{dot+=q.values()[b*256+k]as i32*(fixed.data[index(fixed.rows,cols,start+r,b*256+k)]as i8 as i32);}out[r]+=(dot as f32*q.scales()[b])*sw[r];}}}
 if !crate::all_finite(&out){return Err("K2 single finite".into());}Ok(out)
}

 #[cfg(target_arch="wasm32")]
 #[target_feature(enable="simd128")]
 unsafe fn single_simd(q:&QuantizedRows,w:&PackedView,sw:&[f32],out:&mut[f32],first:usize,end:usize){
  use core::arch::wasm32::*;
  let cols=q.cols();let fixed=&w.fixed;let start=w.start/cols;let stride=fixed.rows/4*cols;
  for r in(0..out.len()).step_by(8){for block in first..end{
   let x=q.values().as_ptr().add(block*256);let p=fixed.data.as_ptr().add(((start+r)/8)*2*cols+block*512);let p1=p.add(stride);let p2=p.add(2*stride);let p3=p.add(3*stride);
   let mut even=i32x4_splat(0);let mut odd=i32x4_splat(0);
   macro_rules! dot2{($k:expr)=>{{let a=v128_load32_splat(x.add($k).cast());let b=v128_load32_splat(x.add($k+128).cast());
    even=i32x4_add(even,i32x4_add(i32x4_dot_i16x8(a,i16x8_load_extend_i8x8(p.add($k*4).cast())),i32x4_dot_i16x8(b,i16x8_load_extend_i8x8(p1.add($k*4).cast()))));
    odd=i32x4_add(odd,i32x4_add(i32x4_dot_i16x8(a,i16x8_load_extend_i8x8(p2.add($k*4).cast())),i32x4_dot_i16x8(b,i16x8_load_extend_i8x8(p3.add($k*4).cast()))));
   }}}
   dot2!(0);dot2!(2);dot2!(4);dot2!(6);dot2!(8);dot2!(10);dot2!(12);dot2!(14);dot2!(16);dot2!(18);dot2!(20);dot2!(22);dot2!(24);dot2!(26);dot2!(28);dot2!(30);dot2!(32);dot2!(34);dot2!(36);dot2!(38);dot2!(40);dot2!(42);dot2!(44);dot2!(46);dot2!(48);dot2!(50);dot2!(52);dot2!(54);dot2!(56);dot2!(58);dot2!(60);dot2!(62);dot2!(64);dot2!(66);dot2!(68);dot2!(70);dot2!(72);dot2!(74);dot2!(76);dot2!(78);dot2!(80);dot2!(82);dot2!(84);dot2!(86);dot2!(88);dot2!(90);dot2!(92);dot2!(94);dot2!(96);dot2!(98);dot2!(100);dot2!(102);dot2!(104);dot2!(106);dot2!(108);dot2!(110);dot2!(112);dot2!(114);dot2!(116);dot2!(118);dot2!(120);dot2!(122);dot2!(124);dot2!(126);
   let lo=i32x4_shuffle::<0,4,1,5>(even,odd);let hi=i32x4_shuffle::<2,6,3,7>(even,odd);let scale=f32x4_splat(q.scales()[block]);
   let a=f32x4_mul(f32x4_mul(f32x4_convert_i32x4(lo),scale),v128_load(sw.as_ptr().add(r).cast()));let b=f32x4_mul(f32x4_mul(f32x4_convert_i32x4(hi),scale),v128_load(sw.as_ptr().add(r+4).cast()));
   v128_store(out.as_mut_ptr().add(r).cast(),f32x4_add(v128_load(out.as_ptr().add(r).cast()),a));v128_store(out.as_mut_ptr().add(r+4).cast(),f32x4_add(v128_load(out.as_ptr().add(r+4).cast()),b));
  }}
 }
