//! Token IDs enter embedding, norm0 and Delta0 without a client round trip.
use crate::{Manifest, Request, Result, prepared_weights::WeightBuffer};
pub(crate) const NAME: &str = "prefix-start-exact-v1";
const C: usize = 2560;
const CONV: usize = 3 * 8192;
const EMBED: &str = "model.language_model.embed_tokens.weight";

fn shape(r: &Request) -> Result<(usize, usize)> {
    if r.encoding != NAME || r.op != "prefix_start_integer" || r.tensor != EMBED
        || r.dims.len() != 2 || !r.aux.is_empty() || !r.scalars.is_empty()
    { return Err("prefix start metadata".into()); }
    let (n, p) = (r.dims[0], r.dims[1]);
    if !(1..=89).contains(&n) || !(1..=132).contains(&p) || n+p>512
    { return Err("prefix start shape".into()); }
    Ok((n,p))
}
pub(crate) fn reply_count(r: &Request) -> Result<usize> {
    let (n,_) = shape(r)?;
    Ok(2*n*C+CONV)
}
pub(crate) fn append_reply(b: &mut Vec<u8>, r: &Request, x: &[f32]) -> Result<()> {
    let (n,_) = shape(r)?;
    if x.len() != reply_count(r)? { return Err("prefix start reply shape".into()); }
    // Each independently bounded block uses the existing exact decoder. No
    // generic float limit changes are needed for their concatenated result.
    b.push(0);
    let length = b.len(); b.extend([0;4]);
    let start = b.len(); crate::block_codec::append(b,&x[..n*C])?;
    let size = b.len()-start; b[length..length+4].copy_from_slice(&(size as u32).to_le_bytes());
    crate::block_codec::append(b,&x[n*C..])
}
pub(crate) fn decode_reply(r: &Request, payload: &[u8]) -> Result<Vec<f32>> {
    let (n,_) = shape(r)?;
    if payload.first()!=Some(&0) || payload.len()<5 {return Err("prefix start reply direction".into());}
    let first = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
    if first<4 || first>payload.len()-5 {return Err("prefix start reply length".into());}
    let mut hidden=crate::block_codec::decode(&payload[5..5+first])?;
    let delta=crate::block_codec::decode(&payload[5+first..])?;
    if hidden.len()!=n*C || delta.len()!=n*C+CONV {return Err("prefix start reply count".into());}
    hidden.extend(delta); Ok(hidden)
}

#[derive(Clone)]
pub struct PreparedPrefixStart { request: Request, ids: Vec<f32>, conv: Vec<f32>, state: crate::delta_full_log::InitialState }
impl PreparedPrefixStart {
    pub(crate) fn decode(r: &Request, payload: &[u8]) -> Result<Self> {
        let (n,p)=shape(r)?;
        let end=5+4*n+2*CONV;
        if payload.first()!=Some(&1) || payload.len()<end+4
            || u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize!=n
        {return Err("prefix start input length/direction".into());}
        let mut ids=Vec::with_capacity(n);
        for bytes in payload[5..5+4*n].chunks_exact(4) {
            let id=u32::from_le_bytes(bytes.try_into().unwrap());
            if id>16_777_216 {return Err("prefix start token range".into());}
            ids.push(id as f32);
        }
        let mut conv=vec![0.;CONV]; crate::bf16_codec::unpack(&payload[5+4*n..end],&mut conv);
        if !conv.iter().all(|v|v.is_finite()) {return Err("prefix start finite conv".into());}
        let length=u32::from_le_bytes(payload[end..end+4].try_into().unwrap()) as usize;
        if !(12..=1_990_000).contains(&length) || payload.len()!=end+4+length
        {return Err("prefix start packet length".into());}
        let packet=&payload[end+4..];
        if &packet[..4]!=b"NPF1" || u32::from_le_bytes(packet[4..8].try_into().unwrap()) as usize!=p
        {return Err("prefix start packet identity".into());}
        let state=crate::prefix_hybrid_codec::decode_for_delta(packet)?;
        Ok(Self{request:r.clone(),ids,conv,state})
    }
    pub(crate) fn evaluate<F,B>(&self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        self.clone().evaluate_owned(r,m,read)
    }
    pub(crate) fn evaluate_owned<F,B>(self,r:&Request,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        if !crate::same_request(r,&self.request) {return Err("prefix start request identity".into());}
        let (n,p)=shape(r)?;
        let mut er=r.clone(); er.op="embed".into(); er.encoding="bf16-block256-exact-v1".into(); er.dims=vec![n,C];
        let(mut hidden,mut bytes)=crate::evaluate_with_prepared_buffer(&er,&self.ids,m,&mut *read)?;
        if hidden.len()!=n*C {return Err("prefix start embedding".into());}
        let mut nr=er.clone(); nr.op="rms_bf16".into(); nr.tensor="model.language_model.layers.0.input_layernorm.weight".into(); nr.scalars=vec![1e-6];
        let t=m.tensors.iter().find(|t|t.name==nr.tensor).ok_or("prefix start norm")?;
        if t.rows!=1 || t.cols!=C || t.dtype!="bf16" || t.bytes!=2*C as u64 {return Err("prefix start norm shape".into());}
        let(w,used)=crate::load_prepared_weight(t,&nr,&mut *read)?; bytes+=used;
        let mut normalized=crate::execute(&nr,&hidden,&w)?; normalized.extend(self.conv);
        let mut dr=er; dr.op="delta_full_log_integer".into(); dr.tensor="model.language_model.layers.0.linear_attn.in_proj_qkv.weight".into(); dr.dims=vec![n,32,p,0];
        let(delta,used)=crate::delta_full_log::evaluate_from_state(&dr,&normalized,m,read,Some(self.state))?; bytes+=used;
        hidden.extend(delta);
        if hidden.len()!=reply_count(r)? || !hidden.iter().all(|v|v.is_finite()) {return Err("prefix start output".into());}
        Ok((hidden,bytes))
    }
}

#[cfg(test)] mod tests {
    use super::*;
    fn request(n:usize)->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"prefix_start_integer","tensor":EMBED,"encoding":NAME,"dims":[n,45],"scalars":[]})).unwrap()}
    #[test] fn bounded_reply_preserves_all_bits_without_relaxing_generic_limit() {
        for n in [1,80,87,89] {
            let r=request(n);let mut x=vec![-0.;reply_count(&r).unwrap()];x[n*C]=1.0000001;
            let frame=crate::encode(&r,&x).unwrap();let(_,y)=crate::decode(&frame).unwrap();
            assert!(x.iter().zip(y).all(|(a,b)|a.to_bits()==b.to_bits()));
            assert!(crate::decode_query(&frame).is_err());
        }
        for n in [0,90,usize::MAX] {assert!(reply_count(&request(n)).is_err());}
        assert!(crate::block_codec::decode(&((crate::MAX_FLOATS+1) as u32).to_le_bytes()).is_err());
    }
    #[test] fn invalid_request_rejects_before_read() {
        let r=request(1); assert!(PreparedPrefixStart::decode(&r,&[]).is_err());
        let mut changed=r;changed.tensor=EMBED.replace("embed_tokens","layers.0");
        assert!(PreparedPrefixStart::decode(&changed,&vec![0;1+4+4+2*CONV+4]).is_err());
    }
    #[test] fn decoded_input_is_bound_and_rejects_invalid_ids_conv_and_packets() {
        let mut r=request(1);r.dims[1]=2;
        let(packet,_)=crate::prefix_hybrid_codec::prepare(&vec![0.;2*6176],2).unwrap();
        let mut payload=vec![1];payload.extend(1u32.to_le_bytes());payload.extend(1u32.to_le_bytes());
        payload.extend(vec![0;2*CONV]);payload.extend((packet.len() as u32).to_le_bytes());payload.extend(packet);
        let input=PreparedPrefixStart::decode(&r,&payload).unwrap();
        assert_eq!(input.ids,vec![1.]);assert_eq!(input.state.value_major().len(),32*128*128);
        let mut changed=r.clone();changed.step=1;
        let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
        assert_eq!(input.evaluate_owned(&changed,&m,&mut|_,_|->Result<Vec<u8>>{panic!("identity read")}).unwrap_err(),"prefix start request identity");
        let mut bad=payload.clone();bad[5..9].copy_from_slice(&16_777_217u32.to_le_bytes());
        assert!(PreparedPrefixStart::decode(&r,&bad).is_err());
        let mut bad=payload.clone();bad[9..11].copy_from_slice(&0x7fc0u16.to_le_bytes());
        assert!(PreparedPrefixStart::decode(&r,&bad).is_err());
        let mut bad=r;bad.dims[1]=3;assert!(PreparedPrefixStart::decode(&bad,&payload).is_err());
        assert!(PreparedPrefixStart::decode(&request(1),&payload[..payload.len()-1]).is_err());
    }
}
