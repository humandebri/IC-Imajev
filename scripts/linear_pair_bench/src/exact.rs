//! Fixed pairs and a query-local duplicated input. No inference state retained.
use imajev_runtime::{Result,int8_kernel::QuantizedRows};
pub struct Prepared{data:Vec<i8>,rows:usize,cols:usize}
pub struct Operands{data:Vec<i16>,rows:usize,cols:usize}
impl Prepared{
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self>{if rows==0||rows%32!=0||cols==0||cols%256!=0||rows.checked_mul(cols)!=Some(w.len())||w.len()>30_000_000{return Err("linear pair weights".into());}let mut data=Vec::with_capacity(w.len());for r in (0..rows).step_by(2){for c in (0..cols).step_by(4){data.extend_from_slice(&w[r*cols+c..r*cols+c+4]);data.extend_from_slice(&w[(r+1)*cols+c..(r+1)*cols+c+4]);}}Ok(Self{data,rows,cols})}
 pub fn bytes(&self)->usize{self.data.len()}
 pub fn project(&self,q:&QuantizedRows,x:&Operands,sw:&[f32],rows:usize,method:u8)->Result<Vec<f32>>{
  if q.cols()!=self.cols||x.cols!=self.cols||x.rows!=q.rows()||q.rows()==0||q.rows()>132||rows==0||rows%32!=0||rows>self.rows||sw.len()!=rows||!sw.iter().all(|v|v.is_finite()&&*v>0.)||rows.checked_mul(q.rows()).is_none_or(|v|v>900000){return Err("linear pair shape".into());}
  if method>3 || (method==3 && (!cfg!(feature="static-87") || q.rows()!=87)){return Err("linear method/rows".into());}
  let mut out=vec![0.;rows*q.rows()];let cols=self.cols;
  for r in (0..rows).step_by(32){let mut sums=vec![[0f32;32];q.rows()];for b in 0..cols/256{
   #[cfg(target_arch="wasm32")]
   // Constructors establish every fixed K256 span and active token row.
   unsafe{if method==3 {#[cfg(feature="static-87")] {let sums:&mut[[f32;32];87]=sums.as_mut_slice().try_into().unwrap();crate::kernel_static::accumulate::<87>(x.data.as_ptr(),self.data.as_ptr().add(r*cols),cols,b*256,q.scales().as_ptr().add(b),cols/256,sw.as_ptr().add(r),sums);} #[cfg(not(feature="static-87"))] {unreachable!();}}else if method==2{crate::kernel::accumulate::<true>(x.data.as_ptr(),self.data.as_ptr().add(r*cols),cols,b*256,q.scales().as_ptr().add(b),cols/256,sw.as_ptr().add(r),&mut sums);}else{crate::kernel::accumulate::<false>(x.data.as_ptr(),self.data.as_ptr().add(r*cols),cols,b*256,q.scales().as_ptr().add(b),cols/256,sw.as_ptr().add(r),&mut sums);}}
   #[cfg(not(target_arch="wasm32"))]
   for i in 0..q.rows(){for j in 0..32{let mut dot=0i32;for c in b*256..(b+1)*256{let wi=(r+j/2*2)*cols+(c/4)*8+(j%2)*4+c%4;dot+=q.values()[i*cols+c]as i32*self.data[wi]as i32;}sums[i][j]+=(dot as f32*q.scales()[i*(cols/256)+b])*sw[r+j];}}
  }for i in 0..q.rows(){out[i*rows+r..i*rows+r+32].copy_from_slice(&sums[i]);}}
  if out.iter().any(|v|!v.is_finite()){return Err("linear pair output".into());}Ok(out)
 }
}
pub fn operands(q:&QuantizedRows)->Operands{let count=q.rows()*q.cols();let mut data:Vec<i16>=Vec::with_capacity(count*2);
 #[cfg(target_arch="wasm32")]
 unsafe{duplicate(q.values().as_ptr(),data.as_mut_ptr(),count);data.set_len(count*2);}
 #[cfg(not(target_arch="wasm32"))]
 for v in q.values()[..count].chunks_exact(4){data.extend_from_slice(v);data.extend_from_slice(v);}
 Operands{data,rows:q.rows(),cols:q.cols()}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn duplicate(input:*const i16,output:*mut i16,count:usize){use core::arch::wasm32::*;for i in(0..count).step_by(4){v128_store(output.add(i*2).cast(),v128_load64_splat(input.add(i).cast()));}}
#[cfg(test)]mod tests{use super::*;
 #[test]fn shapes_extremes_scales(){let rows=32;let cols=512;let w:Vec<_>=(0..rows*cols).map(|i|[-128,127,0,1,-1][i%5]).collect();let f=Prepared::new(&w,rows,cols).unwrap();let sw:Vec<_>=(0..rows).map(|i|0.0123*(i+1)as f32).collect();for n in[1,7,45,80,87,89,132]{let input:Vec<_>=(0..n*cols).map(|i|[-127.,127.,0.,-0.,1.,-1.][i%6]*(1+i/256%3)as f32).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&input,n,cols).unwrap();let expected=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows).unwrap();for method in[1,2]{let v=f.project(&q,&operands(&q),&sw,rows,method).unwrap();assert!(expected.iter().zip(v).all(|(a,b)|a.to_bits()==b.to_bits()));}}}
 #[test]fn invalid_shapes(){assert!(Prepared::new(&[],32,256).is_err());assert!(Prepared::new(&vec![0;32*256],31,256).is_err());}
}
