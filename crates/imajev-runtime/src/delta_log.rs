//! Exact prefix-state representation: shared rounded K, original decay and
//! original F32 innovation. No state quantization or reassociated arithmetic.
use crate::{delta, Result};
const D:usize=128;
fn bounds(n:usize,h:usize,x:&[f32])->Result<()> {
 if n==0 || n>132 || h==0 || h>32 || h%2!=0 || x.len()!=n*(h/2*D+h*D+h) || !x.iter().all(|v|v.is_finite()) {return Err("Delta log shape/finite".into());}
 let decays=&x[n*(h/2*D+h*D)..];
 if !decays.iter().all(|v|(0.0..=1.0).contains(v)) {return Err("Delta log decay".into());}
 Ok(())
}
/// Capture innovations using the unchanged scalar recurrence, beginning at zero.
/// Input uses shared K, raw convolved V, original g followed by original beta.
pub fn capture(n:usize,h:usize,x:&[f32])->Result<(Vec<f32>,Vec<f32>)> {
 if n==0 || n>132 || h==0 || h>32 || h%2!=0 || x.len()!=n*(h/2*D+h*D+2*h) || !x.iter().all(|v|v.is_finite()) {return Err("Delta capture shape/finite".into());}
 let kc=n*h/2*D;let vc=n*h*D;let gates=&x[kc+vc..];
 if !gates.iter().all(|v|(0.0..=1.0).contains(v)) {return Err("Delta capture gates".into());}
 let mut states=vec![0.;h*D*D];let mut innovations=vec![0.;vc];
 for head in 0..h {let s=&mut states[head*D*D..(head+1)*D*D];
  for t in 0..n {let k=&x[(t*h/2+head/2)*D..(t*h/2+head/2+1)*D];let g=gates[t*h+head];let b=gates[n*h+t*h+head];
   for d in 0..D {let row=&mut s[d*D..(d+1)*D];let mut mem=0.;
    for i in 0..D {row[i]*=g;mem+=row[i]*k[i];}
    let update=(x[kc+(t*h+head)*D+d]-mem)*b;
    innovations[(t*h+head)*D+d]=update;
    for i in 0..D {row[i]+=k[i]*update;}
   }
  }
 }
 let mut log=x[..kc].to_vec();log.extend(innovations);log.extend_from_slice(&gates[..n*h]);
 if !log.iter().chain(states.iter()).all(|v|v.is_finite()) {return Err("Delta capture overflow".into());}
 Ok((log,states))
}
/// Reference recurrence state, including computation of unused Q output.
pub fn reference(n:usize,h:usize,x:&[f32])->Result<Vec<f32>> {
 // Share validation with capture without paying for capture in benchmark.
 if n==0 || n>132 || h==0 || h>32 || h%2!=0 || x.len()!=n*(h/2*D+h*D+2*h) || !x.iter().all(|v|v.is_finite()) {return Err("Delta reference shape/finite".into());}
 let kc=n*h/2*D;let vc=n*h*D;let q=vec![0.;n*D];let mut states=vec![0.;h*D*D];
 for head in 0..h {let k:Vec<_>=(0..n).flat_map(|t|x[(t*h/2+head/2)*D..(t*h/2+head/2+1)*D].iter().copied()).collect();
 let v:Vec<_>=(0..n).flat_map(|t|x[kc+(t*h+head)*D..kc+(t*h+head+1)*D].iter().copied()).collect();
 let g:Vec<_>=(0..n).map(|t|x[kc+vc+t*h+head]).collect();let b:Vec<_>=(0..n).map(|t|x[kc+vc+n*h+t*h+head]).collect();
 delta(&q,&k,&v,&g,&b,&mut states[head*D*D..(head+1)*D*D],D,D)?;
 }
 Ok(states)
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
   for i in 0..D {let k=f32x4_splat(*kp.add(i));for d in (0..D).step_by(4) {let p=sp.add(i*D+d);v128_store(p.cast(),f32x4_add(f32x4_mul(v128_load(p.cast()),g),f32x4_mul(k,v128_load(up.add(d).cast()))));}}
  }
  if !KEY_MAJOR {for d in 0..D {for i in 0..D {states[head*D*D+d*D+i]=transposed[i*D+d];}}}
 }states
}

/// Same per-key arithmetic as delta, with the original innovation saved once.
pub(super) fn recorded_head(q:&[f32],k:&[f32],v:&[f32],g:&[f32],beta:&[f32],state:&mut[f32])->Result<(Vec<f32>,Vec<f32>)> {
 let n=g.len();
 if n==0 || n>132 || q.len()!=n*128 || k.len()!=q.len() || v.len()!=q.len() || beta.len()!=n || state.len()!=16384 || !g.iter().chain(beta).all(|v|(0.0..=1.0).contains(v)) {return Err("recorded Delta shape/gates".into());}
 #[cfg(target_arch="wasm32")]
 return Ok(unsafe {crate::delta_recorded_simd::run_recorded(q,k,v,g,beta,state,128,128)});
 #[cfg(not(target_arch="wasm32"))]
 {let mut out=vec![0.;n*128];let mut updates=vec![0.;n*128];for t in 0..n {for d in 0..128 {let row=&mut state[d*128..(d+1)*128];let mut mem=0.;for i in 0..128 {row[i]*=g[t];mem+=row[i]*k[t*128+i];}let update=(v[t*128+d]-mem)*beta[t];updates[t*128+d]=update;let mut value=0.;for i in 0..128 {row[i]+=k[t*128+i]*update;value+=row[i]*q[t*128+i];}out[t*128+d]=value;}}Ok((out,updates))}
}
#[cfg(test)]
mod tests {use super::*;
 #[test] fn recorded_output_and_replayed_state_match_original(){for n in [1,7,45] {let h=2;let mut x:Vec<_>=(0..n*(h/2*D+h*D)).map(|i|((i%17)as f32-8.)/64.).collect();x.extend((0..n*h).map(|i|[0.,1.,0.875][i%3]));x.extend((0..n*h).map(|i|[0.,1.,0.5][i%3]));let(log,s)=capture(n,h,&x).unwrap();let restored=restore(n,h,&log).unwrap();let old=reference(n,h,&x).unwrap();assert!(s.iter().zip(&restored).zip(&old).all(|((a,b),c)|a.to_bits()==b.to_bits()&&a.to_bits()==c.to_bits()));
 let q:Vec<_>=(0..n*128).map(|i|((i%13)as f32-6.)/16.).collect();let k=q.clone();let v=q.clone();let g=vec![0.875;n];let b=vec![0.5;n];let mut a=vec![0.125;16384];let mut c=a.clone();let y=delta(&q,&k,&v,&g,&b,&mut a,128,128).unwrap();let(z,_)=recorded_head(&q,&k,&v,&g,&b,&mut c).unwrap();assert!(a.iter().zip(&c).chain(y.iter().zip(&z)).all(|(a,b)|a.to_bits()==b.to_bits()));}}
 #[test] fn malformed_logs_reject(){for(n,h)in[(0,2),(133,2),(1,0),(1,3),(1,34),(usize::MAX,2)] {assert!(restore(n,h,&[]).is_err());}let mut x=vec![0.;128+256+2];for bad in [f32::NAN,f32::INFINITY,-1.,1.1] {x[384]=bad;assert!(restore(1,2,&x).is_err());}}
}
