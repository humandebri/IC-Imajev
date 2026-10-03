#[cfg(target_arch="wasm32")]
#[path="tile.rs"]
mod tile;
use imajev_runtime::{int8_kernel::QuantizedRows, Result};
pub struct Prepared {data:Vec<i16>,rows:usize,cols:usize}
pub struct Operands {data:Vec<i16>,rows:usize,cols:usize,pairs:usize}
impl Prepared {
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self> {
  if rows==0||rows%32!=0||cols==0||cols%256!=0||rows.checked_mul(cols)!=Some(w.len())||w.len()>30_000_000 {return Err("Strassen16 weights".into());}
  let stride=cols/2;let mut data=vec![0i16;7*(rows/2)*stride];
  for pair in 0..rows/2 {for block in 0..cols/256 {for k in 0..128 {
   let p=pair*2*cols+block*256+k;
   let(b11,b21,b12,b22)=(w[p]as i16,w[p+128]as i16,w[p+cols]as i16,w[p+cols+128]as i16);
   for(m,v)in [b11+b22,b11,b12-b22,b21-b11,b22,b11+b12,b21+b22].into_iter().enumerate(){data[m*(rows/2)*stride+pair*stride+block*128+k]=v;}
  }}}
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize {self.data.len()*2}
 pub fn project(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>> {
  if q.rows()==0||q.rows()>132||q.rows()!=a.rows||q.cols()!=a.cols||q.cols()!=self.cols||rows==0||rows%128!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|s|s.is_finite()&&*s>0.) {return Err("Strassen64 projection".into());}
  #[cfg(target_arch="wasm32")]let out=unsafe{self.simd(q,a,sw,rows)};
  #[cfg(not(target_arch="wasm32"))]let out=self.scalar(q,a,sw,rows);
  if out.iter().any(|v|!v.is_finite()){return Err("Strassen16 finite output".into());}Ok(out)
 }
 #[cfg(not(target_arch="wasm32"))]
 fn scalar(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Vec<f32> {
  let mut out=vec![0.;q.rows()*rows];let stride=self.cols/2;let blocks=self.cols/256;
  for pair in 0..a.pairs {for row in 0..rows/2 {for block in 0..blocks {
   let mut p=[0i32;7];for m in 0..7 {for k in 0..128 {p[m]+=a.data[m*a.pairs*stride+pair*stride+block*128+k]as i32*self.data[m*(self.rows/2)*stride+row*stride+block*128+k]as i32;}}
   let d=[[p[0]+p[3]-p[4]+p[6],p[2]+p[4]],[p[1]+p[3],p[0]-p[1]+p[2]+p[5]]];
   for ti in 0..2 {let t=pair*2+ti;if t<q.rows(){for ri in 0..2 {let r=row*2+ri;out[t*rows+r]+=(d[ti][ri]as f32*q.scales()[t*blocks+block])*sw[r];}}}
  }}}out
 }
 #[cfg(target_arch="wasm32")]
 #[target_feature(enable="simd128")]
 unsafe fn simd(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Vec<f32> {
  let mut out=vec![0.;q.rows()*rows];
  for r in (0..rows).step_by(128) {let mut t=0;
   macro_rules! tile {($n:literal)=>{{tile::evaluate::<$n>(self,q,a,sw,rows,t,r,&mut out);t+=2*$n;}}}
   match q.rows(){45=>tile!(23),80=>{tile!(20);tile!(20);},87=>{tile!(22);tile!(22);},89=>{tile!(23);tile!(22);},132=>{tile!(22);tile!(22);tile!(22);},_=>{
    while t+64<=a.pairs*2{tile!(32);}if t+32<=a.pairs*2{tile!(16);}if t+16<=a.pairs*2{tile!(8);}if t+8<=a.pairs*2{tile!(4);}if t+4<=a.pairs*2{tile!(2);}if t<a.pairs*2{tile!(1);}
   }}debug_assert_eq!(t,a.pairs*2);
  }out
 }

}
pub fn operands(q:&QuantizedRows)->Operands {
 let pairs=q.rows().div_ceil(2);let stride=q.cols()/2;
 #[cfg(target_arch="wasm32")]
 let data={let len=7*pairs*stride;let mut data:Vec<i16>=Vec::with_capacity(len);
  // All complete K128 groups and every product are written before committing
  // the length. Constructor padding supplies the one possible dummy token.
  unsafe{operands_simd(q,pairs,data.as_mut_ptr());data.set_len(len);}data};
 #[cfg(not(target_arch="wasm32"))]
 let data={let mut data=vec![0i16;7*pairs*stride];
 for pair in 0..pairs {for block in 0..q.cols()/256 {for k in 0..128 {
  let p=pair*2*q.cols()+block*256+k;let(a11,a12,a21,a22)=(q.values()[p],q.values()[p+128],q.values()[p+q.cols()],q.values()[p+q.cols()+128]);
  for(m,v)in [a11+a22,a21+a22,a11,a22,a11+a12,a21-a11,a12-a22].into_iter().enumerate(){data[m*pairs*stride+pair*stride+block*128+k]=v;}
 }}}data};Operands{data,rows:q.rows(),cols:q.cols(),pairs}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn operands_simd(q:&QuantizedRows,pairs:usize,out:*mut i16){use core::arch::wasm32::*;
 let cols=q.cols();let stride=cols/2;
 for pair in 0..pairs {for block in 0..cols/256 {
  let p=q.values().as_ptr().add(pair*2*cols+block*256);
  let dst:[*mut i16;7]=core::array::from_fn(|m|out.add(m*pairs*stride+pair*stride+block*128));
  for k in (0..128).step_by(8){
   let a11=v128_load(p.add(k).cast());let a12=v128_load(p.add(128+k).cast());
   let a21=v128_load(p.add(cols+k).cast());let a22=v128_load(p.add(cols+128+k).cast());
   v128_store(dst[0].add(k).cast(),i16x8_add(a11,a22));v128_store(dst[1].add(k).cast(),i16x8_add(a21,a22));
   v128_store(dst[2].add(k).cast(),a11);v128_store(dst[3].add(k).cast(),a22);
   v128_store(dst[4].add(k).cast(),i16x8_add(a11,a12));v128_store(dst[5].add(k).cast(),i16x8_sub(a21,a11));
   v128_store(dst[6].add(k).cast(),i16x8_sub(a12,a22));
  }
 }}
}
#[cfg(test)]mod tests {use super::*;
 #[test]fn row_prefix_uses_full_coefficient_stride(){let cols=512;let rows=256;let w:Vec<i8>=(0..rows*cols).map(|i|((i*17+i/cols*31)%256)as u8 as i8).collect();let sw:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let fixed=Prepared::new(&w,rows,cols).unwrap();
  for n in [87,132]{let x:Vec<f32>=(0..n*cols).map(|i|(i%79)as f32-39.).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let a=operands(&q);let actual=fixed.project(&q,&a,&sw[..128],128).unwrap();let old=imajev_runtime::int8_kernel::project(&q,&w[..128*cols],&sw[..128],128).unwrap();assert!(actual.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()));}
 }
 #[test]fn exact_extremes_scales_and_token_tails(){for cols in [256,512]{let rows=128;let w:Vec<i8>=(0..rows*cols).map(|i|[-128,127,-1,1,0][i%5]).collect();let sw:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let fixed=Prepared::new(&w,rows,cols).unwrap();assert_eq!(fixed.bytes(),w.len()*7/2);
  for n in [1,7,8,32,45,64,80,87,88,89,132]{let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,-0.,0.,0.00123,-41.][i%6]*(1+i/256%5)as f32).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let a=operands(&q);let actual=fixed.project(&q,&a,&sw,rows).unwrap();let old=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows).unwrap();assert!(actual.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()));}
 }}
}
