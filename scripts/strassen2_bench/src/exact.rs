//! Exact two-level factorization. Restore each original block256 integer dot
//! before applying the original F32 scales and original block addition order.
use imajev_runtime::{int8_kernel::QuantizedRows, Result};
#[path="coeff.rs"] mod coeff;
#[cfg(target_arch="wasm32")] #[path="prepare.rs"] mod prepare;
#[cfg(target_arch="wasm32")] #[path="tile.rs"] mod tile;
pub struct Prepared {data:Vec<i16>,rows:usize,cols:usize}
pub struct Operands {data:Vec<i16>,rows:usize,cols:usize,groups:usize}
impl Prepared {
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self> {
  if rows==0||rows%32!=0||cols==0||cols%256!=0||rows.checked_mul(cols)!=Some(w.len())||w.len()>30_000_000 {return Err("Strassen2 weight shape".into());}
  let stride=cols/4;let groups=rows/4;let len=49*groups*stride;
  #[cfg(target_arch="wasm32")]
  let data={let mut data:Vec<i16>=Vec::with_capacity(len);unsafe{prepare::weights(w,rows,cols,data.as_mut_ptr());data.set_len(len);}data};
  #[cfg(not(target_arch="wasm32"))]
  let data={let mut data=vec![0i16;len];for group in 0..groups {for block in 0..cols/256 {for k in 0..64 {for m in 0..49 {
   data[m*groups*stride+group*stride+block*64+k]=coeff::B[m].iter().map(|&(j,c)|c*w[(group*4+j%4)*cols+block*256+j/4*64+k]as i16).sum();
  }}}}data};
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize {self.data.len()*2}
 pub fn project(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>> {
  if q.rows()==0||q.rows()>132||q.rows()!=a.rows||q.cols()!=a.cols||q.cols()!=self.cols||rows==0||rows%128!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|s|s.is_finite()&&*s>0.) {return Err("Strassen2 projection shape".into());}
  #[cfg(target_arch="wasm32")]let out=unsafe{self.simd(q,a,sw,rows)};
  #[cfg(not(target_arch="wasm32"))]let out=self.scalar(q,a,sw,rows);
  if out.iter().any(|v|!v.is_finite()){return Err("Strassen2 finite output".into());}Ok(out)
 }
 #[cfg(not(target_arch="wasm32"))]
 fn scalar(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Vec<f32> {
  let mut out=vec![0.;q.rows()*rows];let stride=self.cols/4;let blocks=self.cols/256;
  for group in 0..a.groups {for row in 0..rows/4 {for block in 0..blocks {
   let mut p=[0i32;49];for m in 0..49 {for k in 0..64 {p[m]+=a.data[m*a.groups*stride+group*stride+block*64+k]as i32*self.data[m*(self.rows/4)*stride+row*stride+block*64+k]as i32;}}
   for ti in 0..4 {let t=group*4+ti;if t<q.rows(){for ri in 0..4 {let r=row*4+ri;let d:i32=coeff::C[ti*4+ri].iter().map(|&(m,c)|c*p[m]).sum();out[t*rows+r]+=(d as f32*q.scales()[t*blocks+block])*sw[r];}}}
  }}}out
 }
 #[cfg(target_arch="wasm32")]
 #[target_feature(enable="simd128")]
 unsafe fn simd(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Vec<f32> {
  let mut out=vec![0.;q.rows()*rows];
  for r in (0..rows).step_by(128) {let mut t=0;
   macro_rules! call {($n:literal)=>{{tile::evaluate::<$n>(self,q,a,sw,rows,t,r,&mut out);t+=4*$n;}}}
   match q.rows(){45=>call!(12),80=>call!(20),87|88=>call!(22),89=>call!(23),132=>{call!(17);call!(16);},_=>{
    if t+128<=a.groups*4{call!(32);}if t+64<=a.groups*4{call!(16);}if t+32<=a.groups*4{call!(8);}if t+16<=a.groups*4{call!(4);}if t+8<=a.groups*4{call!(2);}if t<a.groups*4{call!(1);}
   }}debug_assert_eq!(t,a.groups*4);
  }out
 }
}
pub fn operands(q:&QuantizedRows)->Operands {
 let groups=q.rows().div_ceil(4);let cols=q.cols();let stride=cols/4;let len=49*groups*stride;
 #[cfg(target_arch="wasm32")]
 let data={let mut data:Vec<i16>=Vec::with_capacity(len);unsafe{prepare::inputs(q,groups,data.as_mut_ptr());data.set_len(len);}data};
 #[cfg(not(target_arch="wasm32"))]
 let data={let mut data=vec![0i16;len];for group in 0..groups {for block in 0..cols/256 {for k in 0..64 {for m in 0..49 {
  data[m*groups*stride+group*stride+block*64+k]=coeff::A[m].iter().map(|&(j,c)|c*q.values()[(group*4+j/4)*cols+block*256+j%4*64+k]).sum();
 }}}}data};Operands{data,rows:q.rows(),cols,groups}
}
#[cfg(test)]mod tests {use super::*;
 #[test]fn exact_extremes_scales_and_token_tails(){for cols in [256,512]{let rows=128;let w:Vec<i8>=(0..rows*cols).map(|i|[-128,127,-1,1,0][i%5]).collect();let sw:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let fixed=Prepared::new(&w,rows,cols).unwrap();assert_eq!(fixed.bytes(),w.len()*49/8);
  for n in [1,7,8,32,45,64,80,87,88,89,132]{let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,-0.,0.,0.00123,-41.][i%6]*(1+i/256%5)as f32).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let actual=fixed.project(&q,&operands(&q),&sw,rows).unwrap();let old=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows).unwrap();assert!(actual.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()));}
 }}
 #[test]fn row_prefix_uses_full_coefficient_stride(){let rows=256;let cols=512;let w:Vec<i8>=(0..rows*cols).map(|i|((i*17+i/cols*31)%256)as u8 as i8).collect();let sw:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let fixed=Prepared::new(&w,rows,cols).unwrap();
  for n in [87,132]{let x:Vec<f32>=(0..n*cols).map(|i|(i%79)as f32-39.).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let actual=fixed.project(&q,&operands(&q),&sw[..128],128).unwrap();let old=imajev_runtime::int8_kernel::project(&q,&w[..128*cols],&sw[..128],128).unwrap();assert!(actual.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()));}
 }
}
