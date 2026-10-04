//! Two ordinary queries; exact prepared INT8 input and original F32 A stay client-held.
use crate::{Request,Manifest,Result,prepared_weights::WeightBuffer};
pub(crate) const NAME:&str="mlp-down-state-exact-v1";
pub(crate) fn valid_partial_rows(rows:usize)->bool{rows>0 && rows<2560 && rows%32==0}
const C:usize=2560;const H:usize=9216;const R:usize=64;
pub(crate) fn layout(r:&Request)->Result<(usize,usize,usize,usize)> {
    if r.encoding!=NAME || !(matches!(r.op.as_str(),"mlp_prepare_down"|"mlp_down_norm_prepared") || cfg!(feature="experimental-mlp-delta-fusion") && r.op=="mlp_prepare_partial_down") || !(r.dims.len()==2 || cfg!(feature="experimental-mlp-delta-fusion") && r.op=="mlp_prepare_partial_down" && r.dims.len()==3 && valid_partial_rows(r.dims[2])) || r.dims[0]==0 || r.dims[0]>89 || r.dims[1]!=C || r.scalars.len()!=2 || r.scalars[0].to_bits()!=2f32.to_bits() || r.scalars[1].to_bits()!=1e-6f32.to_bits() || r.aux.len()!=1 {return Err("MLP pipeline metadata".into());}
    let n=r.dims[0];Ok((n,n*C,n*H,n*(H/256+R)))
}
fn root(r:&Request)->Result<(String,String)> {root_scoped(r,false)}
fn root_scoped(r:&Request,allow_final:bool)->Result<(String,String)> {
    let p=r.tensor.strip_suffix(".post_attention_layernorm.weight").ok_or("MLP pipeline norm")?;
    let i=p.strip_prefix("model.language_model.layers.").ok_or("MLP pipeline layer")?;
    let layer:i32=i.parse().map_err(|_|"MLP pipeline layer")?;
    if !(0..32).contains(&layer) || (layer==31 && !allow_final) || i!=layer.to_string() {return Err("MLP pipeline layer".into());}
    let next=if layer==31 {"model.language_model.norm.weight".into()}else{format!("model.language_model.layers.{}.input_layernorm.weight",layer+1)};
    if r.aux[0]!=next {return Err("MLP pipeline next norm".into());}
    Ok((p.to_owned(),next))
}
pub(crate) fn validate_scoped(r:&Request,m:&Manifest,allow_final:bool)->Result<(String,String)> {
    layout(r)?;let(p,next)=root_scoped(r,allow_final)?;
    let specs=[(r.tensor.clone(),1,C,"bf16"),(next.clone(),1,C,"bf16"),
      (format!("{p}.mlp.gate_proj.weight"),H,C,"int8"),(format!("{p}.mlp.up_proj.weight"),H,C,"int8"),(format!("{p}.mlp.down_proj.weight"),C,H,"int8"),
      (format!("{p}.mlp.gate_proj.lora_A.weight"),R,C,"f32"),(format!("{p}.mlp.up_proj.lora_A.weight"),R,C,"f32"),
      (format!("{p}.mlp.gate_proj.lora_B.weight"),H,R,"f32"),(format!("{p}.mlp.up_proj.lora_B.weight"),H,R,"f32"),
      (format!("{p}.mlp.down_proj.lora_A.weight"),R,H,"f32"),(format!("{p}.mlp.down_proj.lora_B.weight"),C,R,"f32")];
    for(name,rows,cols,dtype) in specs {
        let t=m.tensors.iter().find(|t|t.name==name).ok_or("MLP pipeline missing weight")?;
        if t.rows!=rows || t.cols!=cols || t.dtype!=dtype || t.bytes!=if dtype=="f32" {(rows*cols*4)as u64}else if dtype=="bf16" {(rows*cols*2)as u64}else{(rows*cols+rows*4)as u64} {return Err("MLP pipeline weight shape".into());}
    }
    Ok((p,next))
}
pub(crate) fn prepare<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    let(state,bytes)=prepare_internal(r,x,m,read,false)?;
    let n=r.dims[0];let mut flat=Vec::with_capacity(n*(C+H+H/256+R));flat.extend(state.residual);state.q.append_wire_values(&mut flat);flat.extend_from_slice(&state.q.scales()[..n*(H/256)]);flat.extend(state.ax);Ok((flat,bytes))
}
/// Move the first 256 independent down-projection rows into the preparation query.
#[cfg(feature="experimental-mlp-delta-fusion")]
pub(crate) fn prepare_partial<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
 layout(r)?;let rows=*r.dims.get(2).unwrap_or(&256);
 if r.op!="mlp_prepare_partial_down" {return Err("partial MLP prepare op".into());}
 let mut inner=r.clone();inner.op="mlp_prepare_down".into();inner.dims.truncate(2);
 let(mut parts,mut bytes)=prepare_internal(&inner,x,m,read,false)?;
 let n=r.dims[0];let mut dr=inner.clone();dr.op="lora_integer_reuse".into();dr.tensor=format!("{}.mlp.down_proj.weight",parts.root);dr.dims=vec![n,rows,H,0];dr.aux=vec![format!("{}.mlp.down_proj.lora_A.weight",parts.root),format!("{}.mlp.down_proj.lora_B.weight",parts.root)];dr.scalars=vec![2.];
 let(y,used)=crate::projection_reuse::evaluate_prepared(&dr,&parts.q,&parts.ax,m,read)?;bytes+=used;
 for t in 0..n {for d in 0..rows {parts.residual[t*C+d]=crate::bf(parts.residual[t*C+d]+y[t*rows+d]);}}
 let mut flat=Vec::with_capacity(n*(C+H+H/256+R));flat.extend(parts.residual);parts.q.append_wire_values(&mut flat);flat.extend_from_slice(&parts.q.scales()[..n*(H/256)]);flat.extend(parts.ax);Ok((flat,bytes))
}
struct MlpParts{residual:Vec<f32>,q:crate::int8_kernel::QuantizedRows,ax:Vec<f32>,root:String,next:String}
// Created by prepare_internal only after all fixed tensor metadata was checked.
struct VerifiedFinish{root:String,next:String,residual:Vec<f32>}
fn prepare_internal<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F,allow_final:bool)->Result<(MlpParts,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    let(n,count,_,_)=layout(r)?;
    if r.op!="mlp_prepare_down" || x.len()!=2*count || !x.iter().all(|v|v.is_finite()) {return Err("MLP pipeline input".into());}
    let(p,next)=validate_scoped(r,m,allow_final)?;
    let mut nr=r.clone();nr.op="add_norm_bf16".into();nr.dims=vec![n,C];nr.scalars=vec![1e-6];nr.aux.clear();
    let t=m.tensors.iter().find(|t|t.name==r.tensor).unwrap();let(w,rn)=crate::load_prepared_weight(t,&nr,&mut *read)?;
    let mut both=crate::execute(&nr,x,&w)?;
    let mut mr=r.clone();mr.op="mlp_gate_up_integer".into();mr.tensor=format!("{p}.mlp.gate_proj.weight");mr.dims=vec![n,H,C,0];mr.scalars=vec![2.];mr.aux=vec![format!("{p}.mlp.up_proj.weight")];
    let(product,rp)=crate::evaluate_mlp(&mr,&both[count..],m,read)?;
    let q=crate::int8_kernel::quantize_rows(&product,n,H)?;
    let at=m.tensors.iter().find(|t|t.name==format!("{p}.mlp.down_proj.lora_A.weight")).unwrap();
    let mut ar=r.clone();ar.op="matmul".into();ar.tensor=at.name.clone();ar.dims=vec![n,R,H];ar.aux.clear();ar.scalars.clear();
    let(aw,ra)=crate::load_prepared_weight(at,&ar,&mut *read)?;
    let ax=crate::profile::measure("lora_matmul_A_down",||crate::matrix_loaded(&product,&aw,n,R,H))?;
    // Norm has been consumed. Retain the original allocation/residual prefix,
    // rather than copying it into a second temporary before wire export.
    both.truncate(count);
    Ok((MlpParts{residual:both,q,ax,root:p,next},rn+rp+ra))
}
/// Complete small MLP without exporting/reconstructing its prepared state.
#[cfg(feature="experimental-mlp-full")]
pub(crate) fn full<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    if r.op!="mlp_full_integer" || !matches!(r.encoding.as_str(),"bf16-exact"|"bf16-block256-exact-v1") || r.dims.len()!=2 || r.dims[0]==0 || r.dims[0]>if cfg!(feature="experimental-mlp-full89") {89}else{87} {return Err("full MLP metadata/bounds".into());}
    let mut inner=r.clone();inner.encoding=NAME.into();inner.op="mlp_prepare_down".into();
    let(parts,prepared)=prepare_internal(&inner,x,m,read,true)?;
    inner.op="mlp_down_norm_prepared".into();
    let verified=VerifiedFinish{root:parts.root,next:parts.next,residual:parts.residual};
    let state=PreparedMlp{request:inner,residual:Vec::new(),q:parts.q,ax:parts.ax};
    let(y,finished)=state.evaluate_internal(&state.request,m,read,true,Some(verified))?;Ok((y,prepared+finished))
}
/// Independent typed segments are validated once; fields cannot be forged.
/// ```compile_fail
/// let mut state: imajev_runtime::PreparedMlp = todo!();
/// state.ax.clear();
/// ```
pub struct PreparedMlp {request:Request,residual:Vec<f32>,q:crate::int8_kernel::QuantizedRows,ax:Vec<f32>}
impl PreparedMlp {
    pub(crate) fn decode(r:&Request,p:&[u8])->Result<Self> {
        let(n,count,q,tail)=layout(r)?;root(r)?;
        if r.op!="mlp_down_norm_prepared" || p.first()!=Some(&1) || p.len()!=1+count*2+q+tail*4 {return Err("MLP pipeline state length/direction".into());}
        let mut residual=vec![0.;count];crate::bf16_codec::unpack(&p[1..1+count*2],&mut residual);
        let cursor=1+count*2;let floats:Vec<f32>=p[cursor+q..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        if !residual.iter().chain(&floats).all(|v|v.is_finite()) {return Err("MLP pipeline finite state".into());}
        let sx=n*(H/256);let quant=crate::int8_kernel::QuantizedRows::from_bytes(n,H,&p[cursor..cursor+q],&floats[..sx])?;
        Ok(Self{request:r.clone(),residual,q:quant,ax:floats[sx..].to_vec()})
    }
    pub(crate) fn evaluate<F,B>(&self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        self.evaluate_internal(r,m,read,false,None)
    }
    /// Residual slots 0..256 already contain finished BF16 hidden values.
    #[cfg(feature="experimental-mlp-delta-fusion")]
    pub(crate) fn evaluate_partial<F,B>(&self,r:&Request,m:&Manifest,read:&mut F,rows:usize)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        if !valid_partial_rows(rows){return Err("partial MLP rows".into());}
        if !crate::same_request(r,&self.request) {return Err("partial MLP identity".into());}
        let(p,next)=validate_scoped(r,m,false)?;let n=r.dims[0];
        let mut dr=r.clone();dr.op="lora_integer_reuse".into();dr.tensor=format!("{p}.mlp.down_proj.weight");dr.dims=vec![n,C-rows,H,rows];dr.aux=vec![format!("{p}.mlp.down_proj.lora_A.weight"),format!("{p}.mlp.down_proj.lora_B.weight")];dr.scalars=vec![2.];
        let(y,rd)=crate::projection_reuse::evaluate_prepared(&dr,&self.q,&self.ax,m,read)?;
        let mut hidden=self.residual.clone();for t in 0..n {for d in rows..C {hidden[t*C+d]=crate::bf(hidden[t*C+d]+y[t*(C-rows)+d-rows]);}}
        let mut nr=r.clone();nr.op="rms_bf16".into();nr.tensor=next;nr.dims=vec![n,C];nr.scalars=vec![1e-6];nr.aux.clear();
        let t=m.tensors.iter().find(|t|t.name==nr.tensor).ok_or("partial MLP norm")?;let(w,rn)=crate::load_prepared_weight(t,&nr,&mut *read)?;
        let norm=crate::execute(&nr,&hidden,&w)?;hidden.extend(norm);Ok((hidden,rd+rn))
    }
    fn evaluate_internal<F,B>(&self,r:&Request,m:&Manifest,read:&mut F,allow_final:bool,verified:Option<VerifiedFinish>)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        let(p,next,owned_residual)=if let Some(v)=verified {
            // The complete path owns these values and uses self.request directly.
            // No wire/client can supply VerifiedFinish or alter the bound request.
            (v.root,v.next,Some(v.residual))
        }else {
            if serde_json::to_vec(r).map_err(|e|e.to_string())?!=serde_json::to_vec(&self.request).map_err(|e|e.to_string())? {return Err("MLP pipeline identity".into());}
            let(p,next)=validate_scoped(r,m,allow_final)?;(p,next,None)
        };let n=r.dims[0];
        let mut dr=r.clone();dr.op="lora_integer_reuse".into();dr.tensor=format!("{p}.mlp.down_proj.weight");dr.dims=vec![n,C,H,0];dr.aux=vec![format!("{p}.mlp.down_proj.lora_A.weight"),format!("{p}.mlp.down_proj.lora_B.weight")];dr.scalars=vec![2.];
        let(y,rd)=crate::projection_reuse::evaluate_prepared(&dr,&self.q,&self.ax,m,read)?;
        let mut pair=owned_residual.unwrap_or_else(||self.residual.clone());pair.extend(y);
        let mut nr=r.clone();nr.op="add_norm_bf16".into();nr.tensor=next;nr.dims=vec![n,C];nr.scalars=vec![1e-6];nr.aux.clear();
        let t=m.tensors.iter().find(|t|t.name==nr.tensor).unwrap();let(w,rn)=crate::load_prepared_weight(t,&nr,&mut *read)?;
        Ok((crate::execute(&nr,&pair,&w)?,rd+rn))
    }
}
pub(crate) fn append(b:&mut Vec<u8>,r:&Request,x:&[f32])->Result<()> {
    let(_,count,q,tail)=layout(r)?;root(r)?;
    if x.len()==2*count {b.push(0);return crate::block_codec::append(b,x);}
    if x.len()!=count+q+tail || !crate::bf16_codec::all_bf16(&x[..count]) || !x[count+q..count+q+r.dims[0]*(H/256)].iter().all(|v|v.is_finite()&&*v>0.) {return Err("MLP pipeline output shape".into());}
    if b.len()+1+count*2+q+tail*4+32>2_000_000 {return Err("MLP pipeline size".into());}
    b.push(1);let start=b.len();b.resize(start+count*2,0);crate::bf16_codec::pack(&x[..count],&mut b[start..]);
    let start=b.len();b.resize(start+q,0);crate::projection_codec::pack_integers(&x[count..count+q],&mut b[start..])?;
    for v in &x[count+q..] {b.extend_from_slice(&v.to_le_bytes());}Ok(())
}
pub(crate) fn decode_values(r:&Request,p:&[u8])->Result<Vec<f32>> {
    let(_,count,q,tail)=layout(r)?;root(r)?;
    if p.first()==Some(&0) {let v=crate::block_codec::decode(&p[1..])?;if v.len()!=2*count{return Err("MLP pipeline plain shape".into());}return Ok(v);}
    let mut dr=r.clone();dr.op="mlp_down_norm_prepared".into();dr.dims.truncate(2);let state=PreparedMlp::decode(&dr,p)?;
    let mut v=state.residual;v.extend(state.q.values()[..q].iter().map(|v|*v as f32));let sx=r.dims[0]*(H/256);v.extend_from_slice(&state.q.scales()[..sx]);v.extend(state.ax);debug_assert_eq!(v.len(),count+q+tail);Ok(v)
}
#[cfg(test)] mod tests {
 use super::*;
 fn req(n:usize)->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_prepare_down","tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[n,C],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"],"encoding":NAME})).unwrap()}
 #[test] fn invalid_input_fails_before_weight_read() {let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};let mut read=|_,_|->Result<Vec<u8>>{panic!("invalid weight read")};for n in [0,90,usize::MAX]{assert!(prepare(&req(n),&[],&m,&mut read).is_err());}assert!(prepare(&req(1),&vec![0.;2*C],&m,&mut read).is_err());}
 #[cfg(feature="experimental-mlp-full")]
 #[test] fn full_bounds_and_final_layer_scope_reject_before_read(){
 let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};let mut read=|_,_|->Result<Vec<u8>>{panic!("invalid full read")};
 for n in [0,90,usize::MAX]{let mut r=req(n);r.op="mlp_full_integer".into();r.encoding="bf16-block256-exact-v1".into();assert!(full(&r,&[],&m,&mut read).is_err());}
 let mut r=req(1);r.tensor="model.language_model.layers.31.post_attention_layernorm.weight".into();r.aux=vec!["model.language_model.norm.weight".into()];assert!(prepare(&r,&vec![0.;2*C],&m,&mut read).is_err());
 r.op="mlp_full_integer".into();r.encoding="bf16-block256-exact-v1".into();assert!(full(&r,&vec![0.;2*C],&m,&mut read).is_err());
 let mut x=vec![0.;2*C];x[0]=f32::INFINITY;assert!(full(&r,&x,&m,&mut read).is_err());r.encoding=NAME.into();assert!(full(&r,&vec![0.;2*C],&m,&mut read).is_err());
 }
 #[test] fn state_codec_preserves_exact_values_and_rejects_bad_lanes() {
  for n in [1,45,87,89] {let r=req(n);let mut v=vec![-0.;n*C];v.extend((0..n*H).map(|i|[-127.,127.,0.,1.,-1.][i%5]));v.extend(vec![0.00123;n*(H/256)]);v.extend(vec![1.0000001;n*R]);let mut b=vec![];append(&mut b,&r,&v).unwrap();assert!(b.len()<2_000_000);let decoded=decode_values(&r,&b).unwrap();assert!(v.iter().zip(decoded).all(|(a,b)|a.to_bits()==b.to_bits()));let mut dr=r.clone();dr.op="mlp_down_norm_prepared".into();assert!(PreparedMlp::decode(&dr,&b).is_ok());
        let frame=crate::encode(&dr,&v).unwrap();
        assert!(matches!(crate::decode_query(&frame).unwrap().1,crate::DecodedQueryInput::Mlp(_)));
        assert!(crate::decode_query(&crate::encode(&r,&v).unwrap()).is_err());
        let plain=crate::encode(&r,&vec![0.;2*n*C]).unwrap();assert!(crate::decode_query(&plain).is_ok());
        assert!(crate::decode_query(&crate::encode(&dr,&vec![0.;2*n*C]).unwrap()).is_err());
        assert!(PreparedMlp::decode(&dr,&b[..b.len()-1]).is_err());
        let mut invalid=b.clone();let scale=1+n*C*2+n*H;invalid[scale..scale+4].copy_from_slice(&0f32.to_le_bytes());assert!(PreparedMlp::decode(&dr,&invalid).is_err());
        let mut invalid=b.clone();let ax=scale+n*(H/256)*4;invalid[ax..ax+4].copy_from_slice(&f32::NAN.to_le_bytes());assert!(PreparedMlp::decode(&dr,&invalid).is_err());
        b[1+n*C*2]=128;assert!(PreparedMlp::decode(&dr,&b).is_err());}
 }
}
