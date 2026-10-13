#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
pub(crate) unsafe fn rms_ordered_simd(x:&[f32],w:&[f32],eps:f32)->Vec<f32>{
use core::arch::wasm32::*;
let c=x.len();let mut total=0f32;let mut j=0;
while j+4<=c{let v=v128_load(x.as_ptr().add(j).cast());let squares=f32x4_mul(v,v);
total+=f32x4_extract_lane::<0>(squares);total+=f32x4_extract_lane::<1>(squares);total+=f32x4_extract_lane::<2>(squares);total+=f32x4_extract_lane::<3>(squares);j+=4;}
while j<c{let v=x[j];total+=v*v;j+=1;}
let scale=(total/c as f32+eps).sqrt().recip();let sv=f32x4_splat(scale);let mut out=vec![0.;c];j=0;
while j+4<=c{let v=v128_load(x.as_ptr().add(j).cast());let weight=v128_load(w.as_ptr().add(j).cast());v128_store(out.as_mut_ptr().add(j).cast(),f32x4_mul(f32x4_mul(v,sv),weight));j+=4;}
while j<c{out[j]=x[j]*scale*w[j];j+=1;}out
}
