//! Raw INT8 S2: fixed payload unchanged, derived weights shared on first use.
use imajev_runtime::{int8_kernel::QuantizedRows,Result};
#[path="coeff.rs"]mod coeff;
#[cfg(target_arch="wasm32")]#[path="prepare.rs"]mod prepare;
pub struct Prepared{data:Vec<i8>,rows:usize,cols:usize}
pub struct Operands{data:Vec<i16>,rows:usize,cols:usize,groups:usize}
impl Prepared {
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self>{
  if rows==0||rows%32!=0||cols==0||cols%256!=0||rows.checked_mul(cols)!=Some(w.len())||w.len()>30_000_000{return Err("raw S2 weight shape".into());}
  let stride=cols/4;let plane=(rows/4)*stride;let mut data=vec![0i8;w.len()];
  for group in 0..rows/4{for block in 0..cols/256{for k in 0..64{for j in 0..16{data[j*plane+group*stride+block*64+k]=w[(group*4+j%4)*cols+block*256+j/4*64+k];}}}}
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize{self.data.len()}
 pub fn project(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>>{
  if q.rows()==0||q.rows()>132||q.rows()!=a.rows||q.cols()!=a.cols||q.cols()!=self.cols||rows==0||rows%32!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|v|v.is_finite()&&*v>0.){return Err("raw S2 projection shape".into());}
  let mut out=vec![0.;q.rows()*rows];let stride=self.cols/4;let blocks=self.cols/256;
  #[cfg(target_arch="wasm32")]
  unsafe{
   let ap:[*const i16;49]=core::array::from_fn(|m|a.data.as_ptr().add(m*a.groups*stride));
   for r in(0..rows).step_by(32){let wp:[*const i8;16]=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*stride+(r/4)*stride));let mut sums=vec![[0f32;32];q.rows()];
    for block in 0..blocks{crate::s2_kernel::accumulate(ap.as_ptr().cast(),wp.as_ptr().cast(),self.cols,block*256,q.scales().as_ptr().add(block),blocks,sw.as_ptr().add(r),sums.as_mut_ptr().cast(),q.rows());}
    for t in 0..q.rows(){out[t*rows+r..t*rows+r+32].copy_from_slice(&sums[t]);}
   }
  }
  #[cfg(not(target_arch="wasm32"))]
  for group in 0..a.groups{for row in 0..rows/4{for block in 0..blocks{
   let mut p=[0i32;49];for m in 0..49{for k in 0..64{let b:i16=coeff::B[m].iter().map(|&(j,c)|c*self.data[j*(self.rows/4)*stride+row*stride+block*64+k]as i16).sum();p[m]+=a.data[m*a.groups*stride+group*stride+block*64+k]as i32*b as i32;}}
   for ti in 0..4{let t=group*4+ti;if t<q.rows(){for ri in 0..4{let r=row*4+ri;let dot:i32=coeff::C[ti*4+ri].iter().map(|&(m,c)|c*p[m]).sum();out[t*rows+r]+=(dot as f32*q.scales()[t*blocks+block])*sw[r];}}}
  }}}
  if out.iter().any(|v|!v.is_finite()){return Err("raw S2 finite output".into());}Ok(out)
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

#[cfg(test)]mod tests{use super::*;
 #[test]fn exact_raw_payload_and_real_token_boundaries(){let(rows,cols)=(128,512);let w:Vec<i8>=(0..rows*cols).map(|i|[-128,127,0,1,-1][i%5]).collect();let sw:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let fixed=Prepared::new(&w,rows,cols).unwrap();assert_eq!(fixed.bytes(),w.len());
  for n in[1,7,8,32,45,64,80,87,88,89,132]{let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,0.,-0.,1.,-1.,1e-20][i%7]*(1+i/256%3)as f32).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let a=operands(&q);for count in[32,64,128]{let actual=fixed.project(&q,&a,&sw[..count],count).unwrap();let expected=imajev_runtime::int8_kernel::project(&q,&w[..count*cols],&sw[..count],count).unwrap();assert!(actual.iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()),"n={n} rows={count}");}}
 }
}
