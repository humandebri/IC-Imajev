//! Pair-factor fixed correction per K256; original INT8 scales retained.
use imajev_runtime::{Result,int8_kernel::QuantizedRows};
pub struct Prepared{data:Vec<u8>,rows:usize,cols:usize}
pub struct Operands{data:Vec<u8>,rows:usize,cols:usize}
impl Prepared {
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self> {
  if rows==0||rows%32!=0||cols==0||cols%256!=0||rows.checked_mul(cols)!=Some(w.len())||w.len()>30_000_000{return Err("factor weights".into());}
  let mut data=Vec::with_capacity(w.len()+w.len()/64);
  for chunk in w.chunks_exact(256){let mut factor=0i32;for i in 0..128{data.push(chunk[2*i]as u8);factor+=chunk[2*i]as i32*chunk[2*i+1]as i32;}for i in 0..128{data.push(chunk[2*i+1]as u8);}data.extend(factor.to_le_bytes());}
  Ok(Self{data,rows,cols})
 }
 pub fn bytes(&self)->usize{self.data.len()}
 pub fn project(&self,q:&QuantizedRows,x:&Operands,sw:&[f32],rows:usize,method:u8)->Result<Vec<f32>> {
  if q.cols()!=self.cols||x.cols!=self.cols||x.rows!=q.rows()||q.rows()==0||q.rows()>132||rows==0||rows%32!=0||rows>self.rows||sw.len()!=rows||!sw.iter().all(|v|v.is_finite()&&*v>0.)||rows.checked_mul(q.rows()).is_none_or(|v|v>900000)||method!=1{return Err("factor shape".into());}
  let mut out=vec![0.;rows*q.rows()];let blocks=self.cols/256;
  for r in (0..rows).step_by(32){let mut sums=vec![[0f32;32];q.rows()];for b in 0..blocks {
   #[cfg(target_arch="wasm32")]
   // Constructors bound every 516-byte input and 260-byte fixed block.
   unsafe{crate::kernel::accumulate(x.data.as_ptr().cast(),self.data.as_ptr().add(r*blocks*260).cast(),self.cols,b*256,q.scales().as_ptr().add(b),blocks,sw.as_ptr().add(r),sums.as_mut_ptr().cast(),q.rows());}
   #[cfg(not(target_arch="wasm32"))]
   for i in 0..q.rows(){let a=&x.data[(i*blocks+b)*516..(i*blocks+b+1)*516];for j in 0..32{let w=&self.data[((r+j)*blocks+b)*260..((r+j)*blocks+b+1)*260];let mut dot=0i32;for k in 0..128{let ae=i16::from_le_bytes(a[k*2..k*2+2].try_into().unwrap())as i32;let ao=i16::from_le_bytes(a[256+k*2..258+k*2].try_into().unwrap())as i32;dot+=(ae+w[128+k]as i8 as i32)*(ao+w[k]as i8 as i32);}dot-=i32::from_le_bytes(a[512..516].try_into().unwrap())+i32::from_le_bytes(w[256..260].try_into().unwrap());sums[i][j]+=(dot as f32*q.scales()[i*blocks+b])*sw[r+j];}}
  }for i in 0..q.rows(){out[i*rows+r..i*rows+r+32].copy_from_slice(&sums[i]);}}
  if out.iter().any(|v|!v.is_finite()){return Err("factor output".into());}Ok(out)
 }
}
pub fn operands(q:&QuantizedRows)->Operands {
 let mut data=Vec::with_capacity(q.values().len()*2+q.values().len()/64);
 for chunk in q.values().chunks_exact(256){let mut factor=0i32;for i in 0..128{data.extend(chunk[2*i].to_le_bytes());factor+=chunk[2*i]as i32*chunk[2*i+1]as i32;}for i in 0..128{data.extend(chunk[2*i+1].to_le_bytes());}data.extend(factor.to_le_bytes());}
 Operands{data,rows:q.rows(),cols:q.cols()}
}
#[cfg(test)] mod tests {use super::*;
 #[test]fn signed_extremes_scales_and_tails(){let rows=32;let cols=512;let w:Vec<_>=(0..rows*cols).map(|i|[-128,127,0,1,-1][i%5]).collect();let f=Prepared::new(&w,rows,cols).unwrap();let sw:Vec<_>=(0..rows).map(|i|0.0123*(i+1)as f32).collect();for n in [1,7,8,32,45,80,87,89,132]{let v:Vec<_>=(0..n*cols).map(|i|[-127.,127.,0.,-0.,1.,-1.][i%6]*(1+i/256%3)as f32).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&v,n,cols).unwrap();let a=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows).unwrap();let b=f.project(&q,&operands(&q),&sw,rows,1).unwrap();assert!(a.iter().zip(b).all(|(a,b)|a.to_bits()==b.to_bits()));}}
}
