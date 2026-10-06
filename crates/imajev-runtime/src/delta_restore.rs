//! Exact restore shared by the model and the isolated Wasm benchmark.
use crate::Result;
const D:usize=128;
fn bounds(n:usize,h:usize,x:&[f32])->Result<()> {
 if n==0 || n>132 || h==0 || h>32 || h%2!=0 || x.len()!=n*(h/2*D+h*D+h) || !x.iter().all(|v|v.is_finite()) {return Err("Delta log shape/finite".into());}
 let decays=&x[n*(h/2*D+h*D)..];
 if !decays.iter().all(|v|(0.0..=1.0).contains(v)) {return Err("Delta log decay".into());}
 Ok(())
}
pub fn restore(n:usize,h:usize,x:&[f32])->Result<Vec<f32>> { restore_layout::<false>(n,h,x) }
#[cfg(feature="experimental-delta-state-layout")]
pub(crate) fn restore_key_major(n:usize,h:usize,x:&[f32])->Result<Vec<f32>> { restore_layout::<true>(n,h,x) }
fn restore_layout<const KEY_MAJOR:bool>(n:usize,h:usize,x:&[f32])->Result<Vec<f32>> {
 bounds(n,h,x)?;
 #[cfg(target_arch="wasm32")]
 let states=unsafe {restore_simd::<KEY_MAJOR>(n,h,x)};
 #[cfg(not(target_arch="wasm32"))]
 let states={let kc=n*h/2*D;let vc=n*h*D;let mut states=vec![0.;h*D*D];
 for head in 0..h {let s=&mut states[head*D*D..(head+1)*D*D];for t in 0..n {let k=&x[(t*h/2+head/2)*D..(t*h/2+head/2+1)*D];let g=x[kc+vc+t*h+head];for d in 0..D {let update=x[kc+(t*h+head)*D+d];for i in 0..D {let at=if KEY_MAJOR{i*D+d}else{d*D+i};s[at]*=g;s[at]+=k[i]*update;}}}}states};
 if !states.iter().all(|v|v.is_finite()) {return Err("Delta log overflow".into());}Ok(states)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn restore_simd<const KEY_MAJOR:bool>(n:usize,h:usize,x:&[f32])->Vec<f32> {
 use core::arch::wasm32::*;
 let kc=n*h/2*D;let vc=n*h*D;let mut states=vec![0.;h*D*D];
 for head in 0..h {let mut transposed=if KEY_MAJOR{vec![]}else{vec![0f32;D*D]};let sp=if KEY_MAJOR{states.as_mut_ptr().add(head*D*D)}else{transposed.as_mut_ptr()};
  for t in 0..n {let kp=x.as_ptr().add((t*h/2+head/2)*D);let up=x.as_ptr().add(kc+(t*h+head)*D);let g=f32x4_splat(x[kc+vc+t*h+head]);
   #[cfg(not(feature="experimental-prefix-update-hoist"))]
   for i in 0..D {let k=f32x4_splat(*kp.add(i));for d in (0..D).step_by(4) {let p=sp.add(i*D+d);v128_store(p.cast(),f32x4_add(f32x4_mul(v128_load(p.cast()),g),f32x4_mul(k,v128_load(up.add(d).cast()))));}}
   #[cfg(feature="experimental-prefix-update-hoist")]
   {
    // bounds() proves each K/update lane exists. sp belongs to a distinct
    // state allocation. Each state element keeps its original mul/add order.
    // Load the same innovation vectors once per token, not once per key.
    let u0=v128_load(up.add(0).cast());
    let u1=v128_load(up.add(4).cast());
    let u2=v128_load(up.add(8).cast());
    let u3=v128_load(up.add(12).cast());
    let u4=v128_load(up.add(16).cast());
    let u5=v128_load(up.add(20).cast());
    let u6=v128_load(up.add(24).cast());
    let u7=v128_load(up.add(28).cast());
    let u8=v128_load(up.add(32).cast());
    let u9=v128_load(up.add(36).cast());
    let u10=v128_load(up.add(40).cast());
    let u11=v128_load(up.add(44).cast());
    let u12=v128_load(up.add(48).cast());
    let u13=v128_load(up.add(52).cast());
    let u14=v128_load(up.add(56).cast());
    let u15=v128_load(up.add(60).cast());
    let u16=v128_load(up.add(64).cast());
    let u17=v128_load(up.add(68).cast());
    let u18=v128_load(up.add(72).cast());
    let u19=v128_load(up.add(76).cast());
    let u20=v128_load(up.add(80).cast());
    let u21=v128_load(up.add(84).cast());
    let u22=v128_load(up.add(88).cast());
    let u23=v128_load(up.add(92).cast());
    let u24=v128_load(up.add(96).cast());
    let u25=v128_load(up.add(100).cast());
    let u26=v128_load(up.add(104).cast());
    let u27=v128_load(up.add(108).cast());
    let u28=v128_load(up.add(112).cast());
    let u29=v128_load(up.add(116).cast());
    let u30=v128_load(up.add(120).cast());
    let u31=v128_load(up.add(124).cast());
    for i in 0..D {let k=f32x4_splat(*kp.add(i));
     macro_rules! cell {($d:literal,$u:ident)=>{{let p=sp.add(i*D+$d);v128_store(p.cast(),f32x4_add(f32x4_mul(v128_load(p.cast()),g),f32x4_mul(k,$u)));}};}
     cell!(0,u0);
     cell!(4,u1);
     cell!(8,u2);
     cell!(12,u3);
     cell!(16,u4);
     cell!(20,u5);
     cell!(24,u6);
     cell!(28,u7);
     cell!(32,u8);
     cell!(36,u9);
     cell!(40,u10);
     cell!(44,u11);
     cell!(48,u12);
     cell!(52,u13);
     cell!(56,u14);
     cell!(60,u15);
     cell!(64,u16);
     cell!(68,u17);
     cell!(72,u18);
     cell!(76,u19);
     cell!(80,u20);
     cell!(84,u21);
     cell!(88,u22);
     cell!(92,u23);
     cell!(96,u24);
     cell!(100,u25);
     cell!(104,u26);
     cell!(108,u27);
     cell!(112,u28);
     cell!(116,u29);
     cell!(120,u30);
     cell!(124,u31);
    }
   }
  }
  if !KEY_MAJOR {for d in 0..D {for i in 0..D {states[head*D*D+d*D+i]=transposed[i*D+d];}}}
 }states
}

