//! Fixed column/output32 layout with original F32 precision and sum order.
use imajev_runtime::Result;
pub struct Packed { data: Vec<f32>, rows: usize, cols: usize }
impl Packed {
 pub fn new(w:&[f32],rows:usize,cols:usize)->Result<Self>{
  if rows==0||rows%32!=0||cols==0||cols%64!=0||cols>9216||rows.checked_mul(cols)!=Some(w.len())||w.iter().any(|v|!v.is_finite()){return Err("F32 reuse weight shape/finite".into());}
  let mut data=Vec::with_capacity(w.len());
  for tile in (0..rows).step_by(32){for c in 0..cols{for r in 0..32{data.push(w[(tile+r)*cols+c]);}}}
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize{self.data.len()*4}
 pub fn project(&self,x:&[f32],n:usize)->Result<Vec<f32>>{
  if n==0||n>132||n.checked_mul(self.cols)!=Some(x.len())||n*self.rows>900_000||x.iter().any(|v|!v.is_finite()){return Err("F32 reuse input shape/finite".into());}
  let mut out=vec![0.;n*self.rows];
  #[cfg(target_arch="wasm32")]
  for r in(0..self.rows).step_by(32){let mut sums=vec![[0f32;32];n];for start in(0..self.cols).step_by(64){unsafe{crate::kernel::accumulate(x.as_ptr(),self.data.as_ptr().add(r*self.cols),self.cols,start,x.as_ptr(),32,self.data.as_ptr(),sums.as_mut_ptr().cast(),n);}}
   for t in 0..n{out[t*self.rows+r..t*self.rows+r+32].copy_from_slice(&sums[t]);}
  }
  #[cfg(not(target_arch="wasm32"))]
  for t in 0..n{for r in 0..self.rows{for c in 0..self.cols{out[t*self.rows+r]+=x[t*self.cols+c]*self.data[(r/32)*32*self.cols+c*32+r%32];}}}
  if out.iter().any(|v|!v.is_finite()){return Err("F32 reuse output finite".into());}Ok(out)
 }
}
#[cfg(test)]mod tests{use super::*;
 #[test]fn original_precision_order_extremes_and_token_tails(){for rows in[32,64,256]{let cols=64;let w:Vec<_>=(0..rows*cols).map(|i|[0.,-0.,0.00123,-0.234567,1.,-1.][i%6]).collect();let p=Packed::new(&w,rows,cols).unwrap();assert_eq!(p.bytes(),w.len()*4);
 for n in[1,7,8,16,31,32,45,80,87,88,89,132]{let x:Vec<_>=(0..n*cols).map(|i|[-0.,0.,1.,-1.,0.1234567,-0.99999994,f32::MIN_POSITIVE][i%7]).collect();let got=p.project(&x,n).unwrap();let old=imajev_runtime::matrix_reference(&x,&w,n,rows,cols).unwrap();assert!(got.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()),"n={n},rows={rows}");}}
 }
 #[test]fn malformed_weights_and_inputs(){for rows in[0,31,usize::MAX]{assert!(Packed::new(&[],rows,64).is_err());}assert!(Packed::new(&vec![f32::NAN;32*64],32,64).is_err());let p=Packed::new(&vec![1.;32*64],32,64).unwrap();assert!(p.project(&[0.;63],1).is_err());assert!(p.project(&vec![f32::INFINITY;64],1).is_err());assert!(p.project(&[],0).is_err());}
}
