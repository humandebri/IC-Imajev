//! Fixed INT8 weights and a proof byte per output pair / K256 block.
//! No inference state is persisted. Every integer result and F32 order is exact.
use imajev_runtime::{Result,int8_kernel::QuantizedRows};
pub struct Prepared {data:Vec<i8>,modes:Vec<u8>,rows:usize,cols:usize}
pub struct Operands {data:Vec<i8>,rows:usize,cols:usize}
impl Prepared {
 pub fn new(w:&[i8],rows:usize,cols:usize)->Result<Self> {
  if rows==0 || rows%32!=0 || cols==0 || cols%256!=0 || rows.checked_mul(cols)!=Some(w.len()) || w.len()>30_000_000 {return Err("bounded i16 weights".into());}
  let mut data=Vec::with_capacity(w.len());let mut modes=Vec::with_capacity(rows/2*(cols/256));
  for r in (0..rows).step_by(2) {
   for c in (0..cols).step_by(4) {data.extend_from_slice(&w[r*cols+c..r*cols+c+4]);data.extend_from_slice(&w[(r+1)*cols+c..(r+1)*cols+c+4]);}
   for b in 0..cols/256 {
    // Each I16 lane receives one product per K8. For W32 this is
    // four weights per lane, separately for both rows and each subwindow.
    // 127 * 258 = 32766. Bounds on absolute values prove ALL partial sums.
    let safe=(r..r+2).all(|row|(0..8).all(|lane|(0..8).all(|window|
     (0..4).map(|k|(w[row*cols+b*256+window*32+k*8+lane]as i16).unsigned_abs()as u32).sum::<u32>()<=258)));
    modes.push(u8::from(safe));
   }
  }
  Ok(Self{data,modes,rows,cols})
 }
 pub fn bytes(&self)->usize {self.data.len()+self.modes.len()}
 pub fn project(&self,q:&QuantizedRows,x:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>> {
  if q.cols()!=self.cols || x.cols!=self.cols || x.rows!=q.rows() || q.rows()==0 || q.rows()>132 || rows==0 || rows%32!=0 || rows>self.rows || sw.len()!=rows || !sw.iter().all(|v|v.is_finite()&&*v>0.) || rows.checked_mul(q.rows()).is_none_or(|n|n>900000) {return Err("bounded i16 projection".into());}
  let mut out=vec![0.;rows*q.rows()];let cols=self.cols;
  for r in (0..rows).step_by(32) {let mut t=0;
   macro_rules! tile {($n:literal)=>{{let mut sums=[[0f32;32];$n];for b in 0..cols/256 {
    #[cfg(target_arch="wasm32")]
    // Private constructor validates every byte span; no padded token reads.
    unsafe {crate::kernel::accumulate(x.data.as_ptr().add(t*cols*2),self.data.as_ptr().add(r*cols),cols,b*256,q.scales().as_ptr().add(t*(cols/256)+b),cols/256,sw.as_ptr().add(r),self.modes.as_ptr().add(r/2*(cols/256)+b),&mut sums);}
    #[cfg(not(target_arch="wasm32"))]
    for i in 0..$n {for j in 0..32 {let mut d=0i32;for c in b*256..(b+1)*256 {let v=q.values()[(t+i)*cols+c]as i32;let wi=(r+j/2*2)*cols+(c/4)*8+(j%2)*4+c%4;d+=v*self.data[wi]as i32;}sums[i][j]+=(d as f32*q.scales()[(t+i)*(cols/256)+b])*sw[r+j];}}
   }for i in 0..$n {out[(t+i)*rows+r..(t+i)*rows+r+32].copy_from_slice(&sums[i]);}t+=$n;}}}
   match q.rows() {45=>tile!(45),80=>{tile!(40);tile!(40);},87=>{tile!(44);tile!(43);},89=>{tile!(45);tile!(44);},132=>{tile!(44);tile!(44);tile!(44);},_=>{while t+48<=q.rows(){tile!(48);}if t+32<=q.rows(){tile!(32);}if t+16<=q.rows(){tile!(16);}if t+8<=q.rows(){tile!(8);}if t+4<=q.rows(){tile!(4);}if t+2<=q.rows(){tile!(2);}if t<q.rows(){tile!(1);}}}
  }
  if out.iter().any(|v|!v.is_finite()){return Err("bounded i16 output".into());}Ok(out)
 }
}
pub fn operands(q:&QuantizedRows)->Operands {
 let count=q.rows()*q.cols();let mut data:Vec<i8>=Vec::with_capacity(count*2);
 #[cfg(target_arch="wasm32")]
 // Complete K256 rows, initialized opaque I16 lanes in [-127,127], and
 // capacity count*2 prove every narrowing/store and output byte is valid.
 unsafe {pack(q.values().as_ptr(),data.as_mut_ptr(),count);data.set_len(count*2);}
 #[cfg(not(target_arch="wasm32"))]
 for v in q.values()[..count].chunks_exact(4) {for _ in 0..2 {data.extend(v.iter().map(|&v|v as i8));}}
 Operands{data,rows:q.rows(),cols:q.cols()}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn pack(input:*const i16,output:*mut i8,count:usize) {
 use core::arch::wasm32::*;
 for i in (0..count).step_by(16) {
  let a=v128_load(input.add(i).cast());let b=v128_load(input.add(i+8).cast());
  let v=i8x16_narrow_i16x8(a,b);
  v128_store(output.add(i*2).cast(),i8x16_shuffle::<0,1,2,3,0,1,2,3,4,5,6,7,4,5,6,7>(v,v));
  v128_store(output.add(i*2+16).cast(),i8x16_shuffle::<8,9,10,11,8,9,10,11,12,13,14,15,12,13,14,15>(v,v));
 }
}
#[cfg(test)]mod tests {use super::*;
 #[test]fn bounds_cover_extremes_and_original_bits(){let cols=512;let rows=32;for pattern in [0,1,2] {let w:Vec<i8>=(0..rows*cols).map(|i|match pattern {0=>[-128,127,-128,127][i%4],1=>[64,-64,64,-64][i%4],_=>[0,1,-1,20][i%4]}).collect();let f=Prepared::new(&w,rows,cols).unwrap();assert_eq!(f.modes.iter().all(|&v|v==1),pattern!=0);
  // Independent interval proof is checked against the actual packed operands.
  for pair in 0..rows/2 {for b in 0..cols/256 {let win=if f.modes[pair*2+b]==1{32}else{16};for start in (0..256).step_by(win) {for lane in 0..16 {let mut bound=0i32;for k in (start..start+win).step_by(8) {let wi=pair*cols*2+(b*256+k)*2+lane;bound+=(f.data[wi]as i16).unsigned_abs()as i32;}assert!(127*bound<=32767);}}}}
  for n in [1,7,45,87,132] {let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,0.,-0.][i%4]).collect();let q=imajev_runtime::int8_kernel::quantize_rows(&x,n,cols).unwrap();let sw=vec![0.0123;rows];let old=imajev_runtime::int8_kernel::project(&q,&w,&sw,rows).unwrap();let new=f.project(&q,&operands(&q),&sw,rows).unwrap();assert!(old.iter().zip(new).all(|(a,b)|a.to_bits()==b.to_bits()));}
 }}
 #[test]fn reject_bad_shapes_and_scales(){assert!(Prepared::new(&[],32,256).is_err());let f=Prepared::new(&vec![1;32*256],32,256).unwrap();let q=imajev_runtime::int8_kernel::quantize_rows(&vec![1.;256],1,256).unwrap();for v in [0.,-1.,f32::NAN,f32::INFINITY] {assert!(f.project(&q,&operands(&q),&vec![v;32],32).is_err());}}
}
