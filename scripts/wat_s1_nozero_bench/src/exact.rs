use imajev_runtime::{int8_kernel::QuantizedRows, Result};
pub struct Prepared {data:Vec<i8>,rows:usize,cols:usize}
pub struct Operands {data:Vec<i16>,rows:usize,cols:usize,pairs:usize}
impl Prepared {
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self> {
  if rows==0||rows%32!=0||cols==0||cols%256!=0||rows.checked_mul(cols)!=Some(w.len())||w.len()>30_000_000 {return Err("Strassen16 weights".into());}
  let stride=(rows/4)*cols;let mut data=vec![0i8;w.len()];
  for quartet in 0..rows/4 {for pair in 0..2 {for block in 0..cols/256 {for k in 0..128 {
   let row=quartet*4+pair*2;let p=row*cols+block*256+k;
   for(m,v)in [w[p],w[p+128],w[p+cols],w[p+cols+128]].into_iter().enumerate(){data[m*stride+quartet*cols+block*256+(k/4)*8+pair*4+k%4]=v;}
  }}}}
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize {self.data.len()}
 pub fn project(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>> {
  if q.rows()==0||q.rows()>132||q.rows()!=a.rows||q.cols()!=a.cols||q.cols()!=self.cols||rows==0||rows%32!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|s|s.is_finite()&&*s>0.) {return Err("Strassen64 projection".into());}
  #[cfg(target_arch="wasm32")]let out=unsafe{self.simd(q,a,sw,rows)};
  #[cfg(not(target_arch="wasm32"))]let out=self.scalar(q,a,sw,rows);
  if out.iter().any(|v|!v.is_finite()){return Err("Strassen16 finite output".into());}Ok(out)
 }
 #[cfg(not(target_arch="wasm32"))]
 fn coefficient(&self,m:usize,row:usize,block:usize,k:usize)->i16 {
  let stride=(self.rows/4)*self.cols;let index=(row/2)*self.cols+block*256+(k/4)*8+(row%2)*4+k%4;
  let (b11,b21,b12,b22)=(self.data[index]as i16,self.data[stride+index]as i16,self.data[2*stride+index]as i16,self.data[3*stride+index]as i16);
  [b11+b22,b11,b12-b22,b21-b11,b22,b11+b12,b21+b22][m]
 }
 #[cfg(not(target_arch="wasm32"))]
 fn scalar(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Vec<f32> {
  let mut out=vec![0.;q.rows()*rows];let stride=self.cols/2;let blocks=self.cols/256;
  for pair in 0..a.pairs {for row in 0..rows/2 {for block in 0..blocks {
   let mut p=[0i32;7];for m in 0..7 {for k in 0..128 {p[m]+=a.data[m*a.pairs*self.cols+pair*self.cols+block*256+(k/4)*8+k%4]as i32*self.coefficient(m,row,block,k)as i32;}}
   let d=[[p[0]+p[3]-p[4]+p[6],p[2]+p[4]],[p[1]+p[3],p[0]-p[1]+p[2]+p[5]]];
   for ti in 0..2 {let t=pair*2+ti;if t<q.rows(){for ri in 0..2 {let r=row*2+ri;out[t*rows+r]+=(d[ti][ri]as f32*q.scales()[t*blocks+block])*sw[r];}}}
  }}}out
 }
 #[cfg(target_arch="wasm32")]
 unsafe fn simd(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Vec<f32> {
  let mut out=vec![0.;q.rows()*rows];let cols=self.cols;
  let ap:[*const i16;7]=core::array::from_fn(|m|a.data.as_ptr().add(m*a.pairs*cols));
  for r in(0..rows).step_by(32){let wp:[*const i8;4]=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*cols+(r/4)*cols));let mut sums=vec![[0f32;32];q.rows()];
   for b in 0..cols/256{crate::kernel::accumulate(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,b*256,q.scales().as_ptr().add(b),cols/256,sw.as_ptr().add(r),sums.as_mut_ptr().cast(),q.rows());}
   for t in 0..q.rows(){out[t*rows+r..t*rows+r+32].copy_from_slice(&sums[t]);}
  }out
 }

}
pub fn operands(q:&QuantizedRows)->Operands { operands_mode(q,false) }
pub fn operands_mode(q:&QuantizedRows,zero:bool)->Operands {
 #[cfg(not(target_arch="wasm32"))] let _=zero;
 let pairs=q.rows().div_ceil(2);let stride=q.cols()/2;
 #[cfg(target_arch="wasm32")]
 let data={let len=7*pairs*q.cols();let mut data:Vec<i16>=if zero {vec![0;len]}else{Vec::with_capacity(len)};
  // All complete K128 groups and every product are written before committing
  // the length. Constructor padding supplies the one possible dummy token.
  unsafe{operands_simd(q,pairs,data.as_mut_ptr());data.set_len(len);}data};
 #[cfg(not(target_arch="wasm32"))]
 let data={let mut data=vec![0i16;7*pairs*q.cols()];
 for pair in 0..pairs {for block in 0..q.cols()/256 {for k in 0..128 {
  let p=pair*2*q.cols()+block*256+k;let(a11,a12,a21,a22)=(q.values()[p],q.values()[p+128],q.values()[p+q.cols()],q.values()[p+q.cols()+128]);
  for(m,v)in [a11+a22,a21+a22,a11,a22,a11+a12,a21-a11,a12-a22].into_iter().enumerate(){let idx=m*pairs*q.cols()+pair*q.cols()+block*256+(k/4)*8+k%4;data[idx]=v;data[idx+4]=v;}
 }}}data};Operands{data,rows:q.rows(),cols:q.cols(),pairs}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn operands_simd(q:&QuantizedRows,pairs:usize,out:*mut i16){use core::arch::wasm32::*;
 let cols=q.cols();for pair in 0..pairs {for block in 0..cols/256 {
  let p=q.values().as_ptr().add(pair*2*cols+block*256);let dst:[*mut i16;7]=core::array::from_fn(|m|out.add(m*pairs*cols+pair*cols+block*256));
  for k in(0..128).step_by(4){let a11=v128_load64_splat(p.add(k).cast());let a12=v128_load64_splat(p.add(128+k).cast());let a21=v128_load64_splat(p.add(cols+k).cast());let a22=v128_load64_splat(p.add(cols+128+k).cast());
   for(m,v)in[i16x8_add(a11,a22),i16x8_add(a21,a22),a11,a22,i16x8_add(a11,a12),i16x8_sub(a21,a11),i16x8_sub(a12,a22)].into_iter().enumerate(){v128_store(dst[m].add(k*2).cast(),v);}
  }
 }}
}

#[cfg(test)]mod tests {use super::*;
 #[test]fn row_prefix_uses_full_coefficient_stride(){let cols=512;let rows=256;let w:Vec<i8>=(0..rows*cols).map(|i|((i*17+i/cols*31)%256)as u8 as i8).collect();let sw:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let fixed=Prepared::new(&w,rows,cols).unwrap();
  for n in [87,132]{let x:Vec<f32>=(0..n*cols).map(|i|(i%79)as f32-39.).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let a=operands(&q);let actual=fixed.project(&q,&a,&sw[..128],128).unwrap();let old=imajev_runtime::int8_kernel::project(&q,&w[..128*cols],&sw[..128],128).unwrap();assert!(actual.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()));}
 }
 #[test]fn exact_extremes_scales_and_token_tails(){for cols in [256,512]{let rows=32;let w:Vec<i8>=(0..rows*cols).map(|i|[-128,127,-1,1,0][i%5]).collect();let sw:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let fixed=Prepared::new(&w,rows,cols).unwrap();assert_eq!(fixed.bytes(),w.len());
  for n in [1,7,8,32,45,64,80,87,88,89,132]{let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,-0.,0.,0.00123,-41.][i%6]*(1+i/256%5)as f32).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let a=operands(&q);let actual=fixed.project(&q,&a,&sw,rows).unwrap();let old=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows).unwrap();assert!(actual.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()));}
 }}
}

#[cfg(test)]mod raw_tests {use super::*;
 #[test]fn quadrant_storage_is_exact_original_capacity_and_coefficients(){for (rows,cols)in [(32,256),(64,512)] {
  let w:Vec<i8>=(0..rows*cols).map(|i|(i*17+i/cols*31)as u8 as i8).collect();let p=Prepared::new(&w,rows,cols).unwrap();assert_eq!(p.bytes(),w.len());
  for row in 0..rows/2{for b in 0..cols/256{for k in 0..128{let i=row*2*cols+b*256+k;let(a,c,d,e)=(w[i]as i16,w[i+128]as i16,w[i+cols]as i16,w[i+cols+128]as i16);for (m,v)in [a+e,a,d-e,c-a,e,a+d,c+e].into_iter().enumerate(){assert_eq!(p.coefficient(m,row,b,k),v);}}}}
 }for(rows,cols)in [(0,256),(31,256),(32,255),(usize::MAX,256)]{assert!(Prepared::new(&[],rows,cols).is_err());}}
}
