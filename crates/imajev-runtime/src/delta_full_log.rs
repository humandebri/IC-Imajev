//! Full32-head Delta with client-held exact prefix innovations.
use crate::{Manifest,Request,Result,MAX_FLOATS,prepared_weights::WeightBuffer};
const COLS:usize=2560;const H:usize=32;const DK:usize=128;const HISTORY:usize=3*8192;const LOG_ROW:usize=16*128+32*128+32;
/// Layout identity stays attached to decoded state until the recurrence consumes it.
#[derive(Clone)]
pub(crate) enum InitialState {
 ValueMajor(Vec<f32>),
 #[cfg(feature="experimental-delta-state-layout")]
 KeyMajor(Vec<f32>),
}
impl InitialState {
 fn values(&self)->&[f32] {match self {Self::ValueMajor(v)=>v,#[cfg(feature="experimental-delta-state-layout")] Self::KeyMajor(v)=>v}}
 fn into_parts(self)->(Vec<f32>,bool) {match self {Self::ValueMajor(v)=>(v,false),#[cfg(feature="experimental-delta-state-layout")] Self::KeyMajor(v)=>(v,true)}}
 #[cfg(test)]
 pub(crate) fn value_major(&self)->Vec<f32> {let mut v=self.values().to_vec();#[cfg(feature="experimental-delta-state-layout")] if matches!(self,Self::KeyMajor(_)){for h in v.chunks_exact_mut(16384){crate::prefix_hybrid_codec::transpose_head(h);}}v}
}
pub(super) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 evaluate_from_state(r,x,m,read,None)
}
/// A decoded exact packet can supply its state directly; no innovation replay.
pub(super) fn evaluate_from_state<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F,initial:Option<InitialState>)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 evaluate_retaining_state(r,x,m,read,initial,None)
}
/// Server-only continuation retains the exact state after the last token.
pub(super) fn evaluate_retaining_state<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F,initial:Option<InitialState>,retained:Option<&mut Option<InitialState>>)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 if r.op!="delta_full_log_integer" || !crate::lossless_encoding(&r.encoding) || r.dims.len()!=4 || !r.aux.is_empty() || !r.scalars.is_empty() {return Err("full Delta metadata".into());}
 let(n,h,p,keep)=(r.dims[0],r.dims[1],r.dims[2],r.dims[3]);
 if n==0 || n>90 || h!=32 || p>132 || keep>1 || (keep==1 && p!=0) {return Err("full Delta shape".into());}
 if let Some(state)=initial.as_ref() {if p==0 || keep!=0 || state.values().len()!=H*DK*DK || !state.values().iter().all(|v|v.is_finite()) {return Err("full Delta prepared state".into());}}
 let input=n*COLS+HISTORY+if initial.is_some(){0}else{p*LOG_ROW};let output=n*COLS+HISTORY+if keep==1 {n*LOG_ROW}else{0};
 if input>MAX_FLOATS || output>MAX_FLOATS || x.len()!=input || !x.iter().all(|v|v.is_finite()) || !crate::bf16_codec::all_bf16(&x[..n*COLS+HISTORY]) || (initial.is_none() && p>0 && !crate::bf16_codec::all_bf16(&x[n*COLS+HISTORY..n*COLS+HISTORY+p*16*128])) {return Err("full Delta input bounds/precision".into());}
 let bitmap=if r.encoding=="bf16-exact" {output.div_ceil(8)}else{output.div_ceil(256).div_ceil(8)};
 let header=serde_json::to_vec(r).map_err(|e|e.to_string())?.len()+1;
 let reply=4+header+32+4+bitmap+2*(n*COLS+HISTORY)+if keep==1 {2*n*16*128+4*n*(32*128+32)}else{0};
 if reply>2_000_000 {return Err("full Delta reply bounds".into());}
 let root=r.tensor.strip_suffix(".in_proj_qkv.weight").filter(|s|s.ends_with(".linear_attn")).ok_or("full Delta tensor")?;
 let qp=crate::delta_projected::projection(r,m,r.tensor.clone(),n,8192,0)?;
 let zp=crate::delta_projected::projection(r,m,format!("{root}.in_proj_z.weight"),n,4096,0)?;
 let conv_name=format!("{root}.conv1d.weight");let norm_name=format!("{root}.norm.weight");
 let conv=m.tensors.iter().find(|t|t.name==conv_name).ok_or("full Delta conv")?;
 let norm=m.tensors.iter().find(|t|t.name==norm_name).ok_or("full Delta norm")?;
 if conv.rows!=8192 || conv.cols!=4 || norm.rows.checked_mul(norm.cols)!=Some(128) {return Err("full Delta stage weights".into());}
 // Log is validated before expensive projection. No question state is cached.
 let (mut states,key_major)=if let Some(state)=initial {state.into_parts()}else{
  #[cfg(feature="experimental-delta-state-layout")]
  let prepared=if keep==0 {InitialState::KeyMajor(if p==0 {vec![0.;H*DK*DK]}else{crate::delta_log::restore_key_major(p,H,&x[n*COLS+HISTORY..])?})}else{InitialState::ValueMajor(vec![0.;H*DK*DK])};
  #[cfg(not(feature="experimental-delta-state-layout"))]
  let prepared=InitialState::ValueMajor(if p==0 {vec![0.;H*DK*DK]}else{crate::delta_log::restore(p,H,&x[n*COLS+HISTORY..])?});
  prepared.into_parts()
 };
 let original=&x[..n*COLS];let q=crate::profile::measure("activation_quantize",||crate::int8_kernel::quantize_rows(original,n,COLS))?;
 let(qa,mut bytes)=crate::delta_projected::a_product(&qp,original,m,read)?;
 let(za,used)=crate::delta_projected::a_product(&zp,original,m,read)?;bytes+=used;
 let mut gr=r.clone();gr.op="delta_gates_integer".into();gr.tensor=format!("{root}.in_proj_a.weight");gr.dims=vec![n];
 let(gates,used)=crate::delta_stage::gates_cached(&gr,original,m,read,Some(&q))?;bytes+=used;
 let(mixed,used)=crate::evaluate_integer_with_ax(&qp,original,m,read,Some(&q),Some(&qa))?;bytes+=used;
 let(z,used)=crate::evaluate_integer_with_ax(&zp,original,m,read,Some(&q),Some(&za))?;bytes+=used;
 let mut window=x[n*COLS..n*COLS+HISTORY].to_vec();window.extend(mixed);let final_history=window[window.len()-HISTORY..].to_vec();
 let mut cr=r.clone();cr.op="conv_state".into();cr.tensor=conv_name;cr.dims=vec![n,8192,4,0];
 let(w,used)=crate::load_prepared_weight(conv,&cr,&mut *read)?;bytes+=used;
 let convolved=crate::execute(&cr,&window,&w)?;drop(window);drop(w);
 let(nw,used)=crate::load_prepared_weight(norm,r,&mut *read)?;bytes+=used;
 let mut gated=vec![0.;n*4096];let mut saved_k=if keep==1 {vec![0.;n*2048]}else{vec![]};let mut saved_updates=if keep==1 {vec![0.;n*4096]}else{vec![]};
 let mut qh=vec![];let mut kh=vec![];
 for head in 0..32 {
  let gather=|start:usize|->Vec<f32>{(0..n).flat_map(|t|convolved[t*8192+start..t*8192+start+128].iter().copied()).collect()};
  let mut nr=r.clone();nr.op="rms_scaled".into();nr.dims=vec![n,128];nr.aux.clear();nr.scalars=vec![1e-6,1./128.];
  if head%2==0 {qh=crate::execute(&nr,&gather(head/2*128),&[])?;nr.scalars[1]=128f32.sqrt().recip();kh=crate::execute(&nr,&gather(2048+head/2*128),&[])?;
   if keep==1 {for t in 0..n {saved_k[t*2048+head/2*128..t*2048+(head/2+1)*128].copy_from_slice(&kh[t*128..(t+1)*128]);}}
  }
  let vh=gather(4096+head*128);let gh:Vec<_>=(0..n).map(|t|gates[t*32+head]).collect();let bh:Vec<_>=(0..n).map(|t|gates[n*32+t*32+head]).collect();let state=&mut states[head*16384..(head+1)*16384];
  let mut values=if keep==1 {let(y,updates)=crate::delta_log::recorded_head(&qh,&kh,&vh,&gh,&bh,state)?;for t in 0..n {saved_updates[t*4096+head*128..t*4096+(head+1)*128].copy_from_slice(&updates[t*128..(t+1)*128]);}y}else{{
   #[cfg(feature="experimental-delta-state-layout")]
   let y=if key_major {crate::delta_from_key_major(&qh,&kh,&vh,&gh,&bh,state,128,128)?}else if retained.is_some(){crate::delta(&qh,&kh,&vh,&gh,&bh,state,128,128)?}else{crate::delta_without_final_state(&qh,&kh,&vh,&gh,&bh,state,128,128)?};
   #[cfg(all(feature="experimental-delta-no-writeback",not(feature="experimental-delta-state-layout")))]
   let y=if retained.is_some(){crate::delta(&qh,&kh,&vh,&gh,&bh,state,128,128)?}else{crate::delta_without_final_state(&qh,&kh,&vh,&gh,&bh,state,128,128)?};
   #[cfg(not(feature="experimental-delta-no-writeback"))]
   let y=crate::delta(&qh,&kh,&vh,&gh,&bh,state,128,128)?;
   y
  }};
  for v in &mut values {*v=crate::bf(*v);}for t in 0..n {values.extend_from_slice(&z[(t*32+head)*128..(t*32+head+1)*128]);}
  nr.op="gated_norm".into();nr.scalars=vec![1e-6];let y=crate::execute(&nr,&values,&nw)?;for t in 0..n {gated[t*4096+head*128..t*4096+(head+1)*128].copy_from_slice(&y[t*128..(t+1)*128]);}
 }
 let mut op=r.clone();op.op="lora_integer".into();op.tensor=format!("{root}.out_proj.weight");op.dims=vec![n,2560,4096,0];op.scalars=vec![2.];op.aux=vec![format!("{root}.out_proj.lora_A.weight"),format!("{root}.out_proj.lora_B.weight")];
 let(mut out,used)=crate::evaluate_integer_with_ax(&op,&gated,m,read,None,None)?;bytes+=used;
 out.extend(final_history);if keep==1 {out.extend(saved_k);out.extend(saved_updates);out.extend_from_slice(&gates[..n*32]);}
 if out.len()!=output || !out.iter().all(|v|v.is_finite()) {return Err("full Delta output".into());}
 if let Some(slot)=retained {
  #[cfg(feature="experimental-delta-state-layout")]
  { *slot=Some(if key_major {InitialState::KeyMajor(states)} else {InitialState::ValueMajor(states)}); }
  #[cfg(not(feature="experimental-delta-state-layout"))]
  { *slot=Some(InitialState::ValueMajor(states)); }
 }
 Ok((out,bytes))
}
#[cfg(test)]
mod tests {use super::*;
 #[test] fn invalid_shapes_and_reply_bounds_reject_before_reads(){let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"delta_full_log_integer","tensor":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight","dims":[1,32,0,0],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap();for dims in [vec![],vec![0,32,0,0],vec![91,32,0,0],vec![1,16,0,0],vec![1,32,133,0],vec![1,32,1,1],vec![1,32,0,2],vec![usize::MAX,32,0,0]] {r.dims=dims;assert!(evaluate(&r,&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());}r.dims=vec![90,32,0,1];let x=vec![0.;90*2560+HISTORY];assert_eq!(evaluate(&r,&x,&m,&mut|_,_|->Result<Vec<u8>>{panic!("oversize read")}).unwrap_err(),"full Delta reply bounds");}
}
