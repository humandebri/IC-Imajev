//! Synthetic, model-free probes of actual patched Wasm arithmetic.
//! Exported only in the verification build, never in production.
use crate::*;
fn same(a:&[f32],b:&[f32])->u32 {if a.len()!=b.len(){return u32::MAX;}a.iter().zip(b).position(|(a,b)|a.to_bits()!=b.to_bits()).map_or(0,|i|i as u32+1)}
#[no_mangle]
pub extern "C" fn check_integer(n:usize,rows:usize,cols:usize,pattern:usize)->u32 {
 let x:Vec<f32>=(0..n*cols).map(|i|if pattern==0 {0.}else{[-127.,127.,0.,-0.,1.,-1.][(i+i/cols)%6]*(1+i/256%3)as f32}).collect();
 let q=int8_kernel::quantize_rows(&x,n,cols).unwrap();
 let fixed_rows=if pattern<2 {512}else{rows.div_ceil(32)*32};
 let w:Vec<i8>=(0..fixed_rows*cols).map(|i|[-128,127,1,-1,0][(i*17+i/cols*31)%5]).collect();
 let sw:Vec<f32>=(0..fixed_rows).map(|i|0.00123*(i+1)as f32).collect();
 let mut bytes:Vec<u8>=w.iter().map(|&v|v as u8).collect();for v in &sw {bytes.extend(v.to_le_bytes());}
 let fixed=output_pairs::PreparedPairs::from_le_bytes(&bytes,fixed_rows,cols).unwrap();
 let view=fixed.packed_rows(0,rows).unwrap();
 let actual=output_pairs::project(&q,&view,&sw[..rows]).unwrap();
 let mut expected=vec![0.;n*rows];
 for t in 0..n {for r in 0..rows {for block in 0..cols/256 {
  let dot:i32=(block*256..(block+1)*256).map(|c|q.values()[t*cols+c]as i32*w[r*cols+c]as i32).sum();
  expected[t*rows+r]+=(dot as f32*q.scales()[t*(cols/256)+block])*sw[r];
 }}}
 let bad=same(&actual,&expected);if bad!=0{return bad;}
 if fixed_rows<512 {return 0;}
 let mut continued=vec![0.;n*rows];
 for begin in (0..cols).step_by(256) {continued=output_pairs::project_continued(&q,&view,begin,256,&continued).unwrap();}
 same(&continued,&expected)
}
#[no_mangle]
pub extern "C" fn check_float(n:usize,rows:usize,cols:usize)->u32 {
 let x:Vec<f32>=(0..n*cols).map(|i|[-2.,0.,0.125,-0.25,1.][i%5]).collect();
 let w:Vec<f32>=(0..rows*cols).map(|i|[-0.125,0.5,0.,-1.][(i+i/cols)%4]).collect();
 let bytes:Vec<u8>=w.iter().flat_map(|x|x.to_le_bytes()).collect();
 let fixed=prepared_weights::PreparedF32::from_le_bytes_output(&bytes,rows,cols).unwrap();
 let actual=f32_output::project(&x,&fixed,n,rows,cols).unwrap().unwrap();
 let mut expected=vec![0.;n*rows];
 for t in 0..n {for r in 0..rows {for c in 0..cols {expected[t*rows+r]+=x[t*cols+c]*w[r*cols+c];}}}
 let bad=same(&actual,&expected);if bad!=0{return bad;}
 let mut continued=vec![0.;n*rows];
 for begin in (0..cols).step_by(64) {
  let chunk:Vec<f32>=(0..n).flat_map(|t|x[t*cols+begin..t*cols+begin+64].iter().copied()).collect();
  continued=f32_output::continue_columns(&chunk,&continued,&fixed,n,rows,cols,begin,64).unwrap();
 }
 same(&continued,&expected)
}
#[no_mangle]
pub extern "C" fn check_delta(n:usize)->u32 {
 let q:Vec<f32>=(0..n*128).map(|i|((i%13)as f32-6.)/128.).collect();
 let k:Vec<f32>=(0..n*128).map(|i|((i%7)as f32-3.)/64.).collect();
 let v:Vec<f32>=(0..n*128).map(|i|((i%9)as f32-4.)/32.).collect();
 let g=vec![0.875;n];let beta=vec![0.5;n];let mut state=vec![0.125;128*128];
 let mut scalar=state.clone();let mut expected=vec![0.;n*128];
 for t in 0..n {for d in 0..128 {
  let mut mem=0.;for i in 0..128 {scalar[i*128+d]*=g[t];mem+=scalar[i*128+d]*k[t*128+i];}
  let update=(v[t*128+d]-mem)*beta[t];let mut out=0.;
  for i in 0..128 {scalar[i*128+d]+=k[t*128+i]*update;out+=scalar[i*128+d]*q[t*128+i];}
  expected[t*128+d]=out;
 }}
 let actual=unsafe {delta_simd::run::<false,true>(&q,&k,&v,&g,&beta,&mut state,128,128)};
 same(&actual,&expected)
}
#[no_mangle]
pub extern "C" fn check_predicates(n:usize,pattern:usize)->u32 {
 let mut x=vec![1.;n];
 if n>0 {let at=n/2;x[at]=f32::from_bits([0x3f800000,0,0xbf800000,0x7f800000,0xff800000,0x7fc00001,0x80000000,1,0x3f800001,0x7f7fffff][pattern]);}
 let finite=x.iter().all(|v|v.is_finite());let positive=x.iter().all(|v|v.is_finite() && *v>0.);let bf16=x.iter().all(|v|v.to_bits()&65535==0);
 if all_finite(&x)!=finite {return 1;}
 if all_positive_finite(&x)!=positive {return 2;}
 if bf16_codec::all_bf16(&x)!=bf16 {return 3;}
 if bf16_codec::classify_finite(&x).ok()!=if finite {Some(bf16)}else{None} {return 4;}
 let w=vec![0.125;n];
 if n>0 && finite {let sum=x.iter().map(|v|v*v).sum::<f32>();let scale=(sum/n as f32+0.000001).sqrt().recip();let expected:Vec<_>=x.iter().map(|v|v*scale*0.125).collect();if same(&rms(&x,&w,0.000001),&expected)!=0 {return 5;}}
 0
}
#[no_mangle]
pub extern "C" fn check_elementwise(n:usize,c:usize)->u32 {
 prepared_activation::prepare();
 let make=|len:usize,salt:usize|->Vec<f32>{(0..len).map(|i|bf([-2.,-0.,0.,0.125,1.,-0.5][(i+salt)%6])).collect()};
 let a=make(n*c,0);let b=make(n*c,1);
 let expected:Vec<_>=a.iter().zip(&b).map(|(&v,&z)|bf(bf(v)+bf(0.03125*z))).collect();
 if same(&finish_lora_bf16(a.clone(),b.clone(),0.03125),&expected)!=0{return 1;}
 let expected:Vec<_>=a.iter().zip(&b).map(|(&g,&u)|bf(bf_silu(g)*u)).collect();
 if same(&swiglu_owned(a.clone(),b.clone()),&expected)!=0{return 2;}
 let expected:Vec<_>=a.iter().zip(&b).map(|(&v,&g)|bf(bf(v)*silu(g))).collect();
 if same(&gated_norm_owned(a.clone(),&b),&expected)!=0{return 3;}
 let weights=make(c,2);let input=[a,b].concat();let residual:Vec<_>=(0..n*c).map(|i|bf(input[i]+input[n*c+i])).collect();
 let mut expected=residual.clone();for row in residual.chunks_exact(c){let sum=row.iter().map(|v|v*v).sum::<f32>();let scale=(sum/c as f32+0.000001).sqrt().recip();expected.extend(row.iter().zip(&weights).map(|(&v,&w)|bf(v*scale*w)));}
 if same(&add_norm_owned(&input,&weights,n,c,0.000001),&expected)!=0{return 4;}
 let input=make((n+3)*c,3);let weights=make(c*4,4);let mut expected=vec![0.;n*c];
 for t in 0..n {for ch in 0..c {let mut sum=0.;for j in 0..4 {sum+=input[(t+j)*c+ch]*weights[ch*4+j];}expected[t*c+ch]=bf_silu(bf(sum));}}
 expected.extend_from_slice(&input[input.len()-3*c..]);
 if same(&conv_state_owned(&input,&weights,n,c,4),&expected)!=0{return 5;}
 0
}
#[no_mangle]
pub extern "C" fn check_attention_scores(n:usize,p:usize)->u32 {
 let c=256;let q:Vec<_>=(0..n*c).map(|i|bf(((i*17%251)as f32-125.)/128.)).collect();
 let k:Vec<_>=(0..(n+p)*c).map(|i|bf(((i*31%239)as f32-119.)/128.)).collect();
 let mut expected=Vec::new();for t in 0..n {for key in 0..p+t+1 {let mut sum=-0.0f32;for j in 0..c {sum+=q[t*c+j]*k[key*c+j];}expected.push(bf(sum/(c as f32).sqrt()));}}
 same(&attention_key_lanes::scores(&q,&k,n,c,p),&expected)
}
