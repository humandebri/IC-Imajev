//! Finish a client-held partial MLP and the next full attention in one query.
use crate::{prepared_weights::WeightBuffer, Manifest, Request, Result};
pub(crate) const NAME: &str = "mlp-attention-finish-exact-v1";
const C: usize = 2560;
const H: usize = 9216;
const KV: usize = 2048;

fn metadata(r: &Request) -> Result<(usize, usize, usize)> {
    if r.encoding != NAME || !matches!(r.op.as_str(), "mlp_finish_attention_full" | "mlp_finish_attention_full_compact") || r.dims.len() != 3
        || r.aux.len() != 1 || r.scalars.len() != 2
        || r.scalars[0].to_bits() != 2f32.to_bits()
        || r.scalars[1].to_bits() != 1e-6f32.to_bits() {
        return Err("MLP attention finish metadata".into());
    }
    let (n, p, rows) = (r.dims[0], r.dims[1], r.dims[2]);
    if !(1..=89).contains(&n) || p > 132 || !crate::mlp_pipeline::valid_partial_rows(rows) {
        return Err("MLP attention finish bounds".into());
    }
    let text = r.tensor.strip_prefix("model.language_model.layers.")
        .and_then(|v| v.strip_suffix(".post_attention_layernorm.weight"))
        .ok_or("MLP attention finish tensor")?;
    let layer: usize = text.parse().map_err(|_| "MLP attention finish layer")?;
    if layer >= 30 || layer % 4 != 2 || layer.to_string() != text
        || r.aux[0] != format!("model.language_model.layers.{}.input_layernorm.weight", layer + 1) {
        return Err("MLP attention finish scope".into());
    }
    Ok((n, p, layer))
}
fn inner(r: &Request) -> Result<Request> {
    let (n, _, _) = metadata(r)?;
    let mut i = r.clone();
    i.op = "mlp_down_norm_partial_prepared".into();
    i.encoding = crate::mlp_pipeline::NAME.into();
    i.dims = vec![n, C, r.dims[2]];
    Ok(i)
}
pub(crate) fn reply_count(r: &Request) -> Result<usize> {
    let (n, _, _) = metadata(r)?;
    Ok(n * (if r.op == "mlp_finish_attention_full_compact" {2 * C} else {3 * C} + KV))
}
/// Input fields and their complete request binding cannot be forged externally.
/// ```compile_fail
/// let mut state: imajev_runtime::PreparedMlpAttentionFinish = todo!();
/// state.prefix.clear();
/// ```
pub struct PreparedMlpAttentionFinish {
    request: Request,
    mlp_request: Request,
    mlp: crate::PreparedMlp,
    prefix: Vec<f32>,
}
impl PreparedMlpAttentionFinish {
    pub(crate) fn decode(r: &Request, payload: &[u8]) -> Result<Self> {
        let (n, p, _) = metadata(r)?;
        let end = 2 + n * (2 * C + H + 400);
        if payload.first() != Some(&1) || payload.len() != end + 2 * p * KV {
            return Err("MLP attention finish length/direction".into());
        }
        let mlp_request = inner(r)?;
        let mlp = crate::PreparedMlp::decode(&mlp_request, &payload[1..end])?;
        let mut prefix = vec![0.; p * KV];
        crate::bf16_codec::unpack(&payload[end..], &mut prefix);
        if !prefix.iter().all(|v| v.is_finite()) {
            return Err("MLP attention finish prefix finite".into());
        }
        Ok(Self { request: r.clone(), mlp_request, mlp, prefix })
    }
    pub(crate) fn evaluate<F, B>(self, r: &Request, m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
    where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer {
        if !crate::same_request(r, &self.request) {
            return Err("MLP attention finish identity".into());
        }
        let (n, p, layer) = metadata(r)?;
        let (mut both, used_mlp) = crate::profile::measure("bridge_mlp_finish", ||
            self.mlp.evaluate(&self.mlp_request, m, read))?;
        if both.len() != 2 * n * C {return Err("MLP attention finish MLP shape".into());}
        let mut input = both[n * C..].to_vec();
        input.extend(self.prefix);
        let mut ar = r.clone();
        ar.op = "attention_full_integer".into();
        ar.encoding = "bf16-block256-exact-v1".into();
        ar.tensor = format!("model.language_model.layers.{}.self_attn.q_proj.weight", layer + 1);
        ar.dims = vec![n, p, 0]; ar.aux.clear(); ar.scalars.clear();
        let (attention, used_attention) = crate::profile::measure("bridge_attention_full", ||
            crate::attention_full::evaluate(&ar, &input, m, read))?;
        if attention.len() != n * (C + KV) {return Err("MLP attention finish attention shape".into());}
        if r.op == "mlp_finish_attention_full_compact" {both.truncate(n * C);}
        both.extend(attention);
        Ok((both, used_mlp.checked_add(used_attention).ok_or("MLP attention finish read overflow")?))
    }
}
pub(crate) fn append_reply(out: &mut Vec<u8>, r: &Request, values: &[f32]) -> Result<()> {
    if values.len() != reply_count(r)? || !crate::bf16_codec::classify_finite(values)? {
        return Err("MLP attention finish reply shape/precision".into());
    }
    out.push(0); let start = out.len(); out.resize(start + 2 * values.len(), 0);
    crate::bf16_codec::pack(values, &mut out[start..]); Ok(())
}
pub(crate) fn decode_reply(r: &Request, payload: &[u8]) -> Result<Vec<f32>> {
    let count = reply_count(r)?;
    if payload.first() != Some(&0) || payload.len() != 1 + 2 * count {
        return Err("MLP attention finish reply length/direction".into());
    }
    let mut out = vec![0.; count]; crate::bf16_codec::unpack(&payload[1..], &mut out);
    if !out.iter().all(|v| v.is_finite()) {return Err("MLP attention finish reply finite".into());}
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(n:usize)->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_finish_attention_full","encoding":NAME,"tensor":"model.language_model.layers.6.post_attention_layernorm.weight","dims":[n,45,1280],"scalars":[2.,1e-6],"aux":["model.language_model.layers.7.input_layernorm.weight"]})).unwrap()}
    fn payload(r:&Request)->Vec<u8> {
        let n=r.dims[0];let mut v=vec![-0.;n*C];v.extend(vec![127.;n*H]);v.extend(vec![0.0123;n*36]);v.extend(vec![0.1234567;n*64]);
        let mut b=vec![1];crate::mlp_pipeline::append(&mut b,&inner(r).unwrap(),&v).unwrap();b.extend(vec![0;2*r.dims[1]*KV]);b
    }
    #[test]fn codec_and_direction_preserve_bits_and_frame_bounds() {
        for n in [1,7,87,89] {let r=request(n);let p=payload(&r);assert!(p.len()+16424<2_000_000);assert!(PreparedMlpAttentionFinish::decode(&r,&p).is_ok());
            let v=vec![-0.;reply_count(&r).unwrap()];let frame=crate::encode(&r,&v).unwrap();let(_,out)=crate::decode(&frame).unwrap();assert!(v.iter().zip(out).all(|(a,b)|a.to_bits()==b.to_bits()));assert!(frame.len()<2_000_000);
            assert!(crate::decode_query(&frame).is_err());
        }
    }
    #[test]fn compact_reply_omits_only_unused_norm_and_keeps_legacy_contract() {
        for n in [1,87,89] {
            let mut r=request(n);let legacy=reply_count(&r).unwrap();r.op="mlp_finish_attention_full_compact".into();
            assert_eq!(legacy-reply_count(&r).unwrap(),n*C);
            assert!(PreparedMlpAttentionFinish::decode(&r,&payload(&r)).is_ok());
            let v=vec![-0.;reply_count(&r).unwrap()];let frame=crate::encode(&r,&v).unwrap();
            assert!(crate::decode(&frame).unwrap().1.iter().all(|x|x.to_bits()==(-0f32).to_bits()));
            assert!(crate::encode(&r,&vec![0.;legacy]).is_err());
        }
    }
    #[test]fn malformed_input_and_all_identity_fields_fail_before_reads() {
        let r=request(1);let p=payload(&r);
        for dims in [vec![],vec![0,45,1280],vec![90,45,1280],vec![1,133,1280],vec![1,45,0],vec![1,45,C],vec![1,45,31],vec![usize::MAX,45,1280]] {let mut bad=r.clone();bad.dims=dims;assert!(PreparedMlpAttentionFinish::decode(&bad,&p).is_err());}
        for index in [0,1,2+2*C] {let mut bad=p.clone();bad[index]=128;assert!(PreparedMlpAttentionFinish::decode(&r,&bad).is_err());}
        let mut bad=p.clone();let end=bad.len();bad[end-2..].copy_from_slice(&0x7fc0u16.to_le_bytes());assert!(PreparedMlpAttentionFinish::decode(&r,&bad).is_err());
        assert!(PreparedMlpAttentionFinish::decode(&r,&p[..p.len()-1]).is_err());bad=p.clone();bad.push(0);assert!(PreparedMlpAttentionFinish::decode(&r,&bad).is_err());
        let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
        for field in 0..11 {let mut bad=r.clone();match field {0=>bad.version+=1,1=>bad.step+=1,2=>bad.model="d".repeat(64),3=>bad.pack_hash="d".repeat(64),4=>bad.input_hash="d".repeat(64),5=>bad.op="bad".into(),6=>bad.encoding="bad".into(),7=>bad.tensor="bad".into(),8=>bad.aux.push("bad".into()),9=>bad.dims[2]=1408,_=>bad.scalars[0]=1.};
            let state=PreparedMlpAttentionFinish::decode(&r,&p).unwrap();assert_eq!(state.evaluate(&bad,&m,&mut|_,_|->Result<Vec<u8>>{panic!("identity read")}).err().unwrap(),"MLP attention finish identity");
        }
    }
}
