use crate::*;
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn finish_lora_simd(base:&mut[f32],z:&[f32],scale:f32){
 use core::arch::wasm32::*;
 #[inline(always)]unsafe fn round(v:v128)->v128{
  let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));
  v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))
 }
 let mut i=0;let s=f32x4_splat(scale);
 while i+4<=base.len(){
  let b=round(v128_load(base.as_ptr().add(i).cast()));
  let l=round(f32x4_mul(s,v128_load(z.as_ptr().add(i).cast())));
  v128_store(base.as_mut_ptr().add(i).cast(),round(f32x4_add(b,l)));i+=4;
 }
 while i<base.len(){base[i]=bf(bf(base[i])+bf(scale*z[i]));i+=1;}
}

pub(crate) fn finish_lora_bf16(mut base:Vec<f32>,z:Vec<f32>,scale:f32)->Vec<f32>{
 assert_eq!(base.len(),z.len());
 #[cfg(target_arch="wasm32")]
 if base.len()>=4 {
  // Both vectors contain initialized F32 elements; four-lane spans are in bounds.
  // Wasm SIMD is available in this runtime. Separate multiply/add and each BF16
  // round retain the original F32 operation order, including scalar tails.
  unsafe{finish_lora_simd(&mut base,&z,scale)};return base;
 }
 base.into_iter().zip(z).map(|(v,z)|bf(bf(v)+bf(scale*z))).collect()
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub unsafe fn conv4_simd<const ACT:bool>(x:&[f32],weight:&[f32],n:usize,c:usize)->Vec<f32>{
 use core::arch::wasm32::*;
 assert!(n>0&&c>0&&c%4==0&&x.len()==(n+3)*c&&weight.len()==c*4);
 let mut y=vec![0.;n*c];
 let run=|table:Option<&[f32;65536]>|{
 let emit=|v|if ACT{
  let b=bf(v);if b.is_finite(){if let Some(t)=table{return t[(b.to_bits()>>16)as usize];}}
  bf_silu_original(b)
 }else{v};
 for ch in (0..c).step_by(4){
  let a=v128_load(weight.as_ptr().add(ch*4).cast());let b=v128_load(weight.as_ptr().add((ch+1)*4).cast());
  let c0=v128_load(weight.as_ptr().add((ch+2)*4).cast());let d=v128_load(weight.as_ptr().add((ch+3)*4).cast());
  let lo0=i32x4_shuffle::<0,4,1,5>(a,b);let lo1=i32x4_shuffle::<0,4,1,5>(c0,d);
  let hi0=i32x4_shuffle::<2,6,3,7>(a,b);let hi1=i32x4_shuffle::<2,6,3,7>(c0,d);
  let w0=i32x4_shuffle::<0,1,4,5>(lo0,lo1);let w1=i32x4_shuffle::<2,3,6,7>(lo0,lo1);
  let w2=i32x4_shuffle::<0,1,4,5>(hi0,hi1);let w3=i32x4_shuffle::<2,3,6,7>(hi0,hi1);
  for t in 0..n{
   let mut sum=f32x4_splat(0.);
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add(t*c+ch).cast()),w0));
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add((t+1)*c+ch).cast()),w1));
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add((t+2)*c+ch).cast()),w2));
   sum=f32x4_add(sum,f32x4_mul(v128_load(x.as_ptr().add((t+3)*c+ch).cast()),w3));
   let p=y.as_mut_ptr().add(t*c+ch);
   *p=emit(f32x4_extract_lane::<0>(sum));*p.add(1)=emit(f32x4_extract_lane::<1>(sum));
   *p.add(2)=emit(f32x4_extract_lane::<2>(sum));*p.add(3)=emit(f32x4_extract_lane::<3>(sum));
  }
 }
 y.extend_from_slice(&x[x.len()-3*c..]);y
 };
 if ACT{prepared_activation::with_table::<1,_>(run)}else{run(None)}
}

#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub(crate) unsafe fn add_norm_simd(x:&[f32],w:&[f32],n:usize,c:usize,eps:f32)->Vec<f32>{
use core::arch::wasm32::*;
#[inline(always)]unsafe fn round(v:v128)->v128{let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))}
let count=n*c;let mut out=vec![0.;count*2];let mut i=0;
while i+4<=count{v128_store(out.as_mut_ptr().add(i).cast(),round(f32x4_add(v128_load(x.as_ptr().add(i).cast()),v128_load(x.as_ptr().add(count+i).cast()))));i+=4;}
while i<count{out[i]=bf(x[i]+x[count+i]);i+=1;}
for row in 0..n {
let start=row*c;let mut total=0f32;let mut j=0;
while j+4<=c{let v=v128_load(out.as_ptr().add(start+j).cast());let squares=f32x4_mul(v,v);
total+=f32x4_extract_lane::<0>(squares);total+=f32x4_extract_lane::<1>(squares);total+=f32x4_extract_lane::<2>(squares);total+=f32x4_extract_lane::<3>(squares);j+=4;}
while j<c{let v=out[start+j];total+=v*v;j+=1;}
let scale=(total/c as f32+eps).sqrt().recip();let sv=f32x4_splat(scale);j=0;
while j+4<=c{let v=v128_load(out.as_ptr().add(start+j).cast());let weight=v128_load(w.as_ptr().add(j).cast());v128_store(out.as_mut_ptr().add(count+start+j).cast(),round(f32x4_mul(f32x4_mul(v,sv),weight)));j+=4;}
while j<c{out[count+start+j]=bf(out[start+j]*scale*w[j]);j+=1;}
}out
}


pub fn swiglu_scalar(gate:Vec<f32>,up:Vec<f32>)->Vec<f32>{gate.into_iter().zip(up).map(|(g,u)|bf(bf_silu(g)*u)).collect()}
pub fn swiglu_owned(mut gate:Vec<f32>,up:Vec<f32>)->Vec<f32>{
 assert_eq!(gate.len(),up.len());
 #[cfg(target_arch="wasm32")]
 if gate.len()>=4{prepared_activation::with_table::<1,_>(|table|unsafe{swiglu_simd(&mut gate,&up,table)});return gate;}
 swiglu_scalar(gate,up)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn swiglu_simd(gate:&mut[f32],up:&[f32],table:Option<&[f32;65536]>){
use core::arch::wasm32::*;
#[inline(always)]unsafe fn round(v:v128)->v128{let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))}
let emit=|g:f32|{let bits=g.to_bits();if bits&65535==0 && g.is_finite(){if let Some(t)=table{return t[(bits>>16)as usize];}}bf_silu_original(g)};
let mut i=0;
while i+4<=gate.len(){let g=gate.as_mut_ptr().add(i);let v=f32x4(emit(*g),emit(*g.add(1)),emit(*g.add(2)),emit(*g.add(3)));let y=round(f32x4_mul(v,v128_load(up.as_ptr().add(i).cast())));v128_store(g.cast(),y);i+=4;}
while i<gate.len(){gate[i]=bf(emit(gate[i])*up[i]);i+=1;}
}

pub fn gated_norm_scalar(values:Vec<f32>,gate:&[f32])->Vec<f32>{
    values.into_iter().zip(gate.iter().copied()).map(|(v,g)|bf(bf(v)*silu(g))).collect()
}
pub fn gated_norm_owned(mut values:Vec<f32>,gate:&[f32])->Vec<f32>{
    assert_eq!(values.len(),gate.len());
    #[cfg(target_arch="wasm32")]
    {prepared_activation::with_table::<3,_>(|table|unsafe{gated_norm_simd(&mut values,&gate,table)});return values;}
    #[cfg(not(target_arch="wasm32"))]
    gated_norm_scalar(values,gate)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn gated_norm_simd(values:&mut[f32],gate:&[f32],table:Option<&[f32;65536]>){
    use core::arch::wasm32::*;
    #[inline(always)]unsafe fn round(v:v128)->v128{let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))}
    let emit=|g:f32|{let bits=g.to_bits();if bits&65535==0 && g.is_finite(){if let Some(t)=table{return t[(bits>>16)as usize];}}silu_original(g)};
    let mut i=0;
    while i+4<=values.len(){let p=values.as_mut_ptr().add(i);let g=gate.as_ptr().add(i);let v=round(v128_load(p.cast()));let s=f32x4(emit(*g),emit(*g.add(1)),emit(*g.add(2)),emit(*g.add(3)));v128_store(p.cast(),round(f32x4_mul(v,s)));i+=4;}
    while i<values.len(){values[i]=bf(bf(values[i])*emit(gate[i]));i+=1;}
}
