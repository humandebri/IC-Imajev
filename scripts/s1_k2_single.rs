impl Prepared {
 pub fn project_single(&self,q:&QuantizedRows,sw:&[f32],rows:usize)->Result<Vec<f32>> {
  if q.rows()!=1||q.cols()!=self.cols||rows==0||rows%8!=0||rows>self.rows||sw.len()!=rows||!sw.iter().all(|v|v.is_finite()&&*v>0.){return Err("K2 single shape".into());}
  let mut out=vec![0f32;rows];
  #[cfg(target_arch="wasm32")]unsafe{self.single_simd(q,sw,&mut out);}
  #[cfg(not(target_arch="wasm32"))]
  for r in 0..rows{for block in 0..self.cols/256{let mut dot=0i32;for k in 0..256{let m=k/128+(r%2)*2;let index=m*(self.rows/4)*self.cols+(r/8)*2*self.cols+block*512+((k%128)/2)*8+((r%8)/2)*2+k%2;dot+=q.values()[block*256+k]as i32*self.data[index]as i32;}out[r]+=(dot as f32*q.scales()[block])*sw[r];}}
  if !imajev_runtime::all_finite(&out){return Err("K2 single finite".into());}Ok(out)
 }
 #[cfg(target_arch="wasm32")]
 #[target_feature(enable="simd128")]
 unsafe fn single_simd(&self,q:&QuantizedRows,sw:&[f32],out:&mut[f32]){
  use core::arch::wasm32::*;
  let cols=self.cols;let stride=self.rows/4*cols;
  for r in(0..out.len()).step_by(8){for block in 0..cols/256{
   let x=q.values().as_ptr().add(block*256);let p=self.data.as_ptr().add((r/8)*2*cols+block*512);let p1=p.add(stride);let p2=p.add(2*stride);let p3=p.add(3*stride);
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
}
