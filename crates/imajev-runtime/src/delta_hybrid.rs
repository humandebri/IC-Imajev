//! Exact client-held prefix packet consumed directly by the full Delta path.
use crate::{Manifest, Request, Result, prepared_weights::WeightBuffer};
pub(crate) const NAME: &str = "delta-hybrid-prefix-exact-v1";

/// Fields are constructed only after validating the envelope and exact packet.
pub struct PreparedDeltaHybrid {
    request: Request,
    input: Vec<f32>,
    state: crate::delta_full_log::InitialState,
}

/// A job owns this carry. It never mutates the immutable prefix cache.
#[cfg(feature="experimental-update-token-chunks")]
pub struct ServerDeltaStream {
    bound: Request,
    state: Option<crate::delta_full_log::InitialState>,
    next: usize,
}

#[cfg(feature="experimental-update-token-chunks")]
impl ServerDeltaStream {
    pub(crate) fn new(input: PreparedDeltaHybrid) -> Self {
        Self { bound: input.request, state: Some(input.state), next: 0 }
    }

    pub fn evaluate<F,B>(&mut self, r:&Request, x:&[f32], offset:usize,
                         m:&Manifest, read:&mut F)->Result<(Vec<f32>,u64)>
    where F:FnMut(u64,usize)->Result<B>, B:WeightBuffer {
        let b=&self.bound;
        if offset!=self.next || r.version!=b.version || r.model!=b.model || r.pack_hash!=b.pack_hash
            || r.input_hash!=b.input_hash || r.tensor!=b.tensor || r.op!=b.op || r.encoding!=b.encoding
            || !r.aux.is_empty() || !r.scalars.is_empty() || r.dims.len()!=4
            || !(1..=if cfg!(feature="experimental-adaptive-token-tiles"){89}else{57}).contains(&r.dims[0]) || r.dims[1..]!=b.dims[1..]
            || offset.checked_add(r.dims[0]).is_none_or(|end|end>485)
            || self.state.is_none() || m.model!=r.model || m.pack_hash!=r.pack_hash {
            return Err("server Delta stream identity/progress".into());
        }
        // The job is failed and released if evaluation errors. A consumed state
        // cannot silently restart from the fixed prefix after a partial failure.
        let initial=self.state.take();
        let mut inner=r.clone();inner.op="delta_full_log_integer".into();
        inner.encoding="bf16-block256-exact-v1".into();
        let result=crate::delta_full_log::evaluate_retaining_state(&inner,x,m,read,initial,Some(&mut self.state))?;
        self.next+=r.dims[0];
        Ok(result)
    }
}

impl PreparedDeltaHybrid {
    pub(crate) fn decode(r: &Request, payload: &[u8]) -> Result<Self> {
        if r.encoding != NAME || r.op != "delta_full_hybrid_integer"
            || r.dims.len() != 4 || !r.aux.is_empty() || !r.scalars.is_empty()
            || !(1..=90).contains(&r.dims[0]) || r.dims[1] != 32
            || !(1..=132).contains(&r.dims[2]) || r.dims[3] != 0
        { return Err("hybrid Delta metadata".into()); }
        let count = r.dims[0] * 2560 + 3 * 8192;
        let end = 5 + count * 2;
        if payload.first() != Some(&1) || payload.len() < end + 4
            || u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize != count
        { return Err("hybrid Delta input length/direction".into()); }
        let packet_len = u32::from_le_bytes(payload[end..end+4].try_into().unwrap()) as usize;
        if packet_len > 1_990_000 || packet_len < 12 || payload.len() != end + 4 + packet_len
        { return Err("hybrid Delta packet length".into()); }
        let packet = &payload[end+4..];
        if &packet[..4] != b"NPF1" || u32::from_le_bytes(packet[4..8].try_into().unwrap()) as usize != r.dims[2]
        { return Err("hybrid Delta prefix identity".into()); }
        let mut input = vec![0.; count];
        crate::bf16_codec::unpack(&payload[5..end], &mut input);
        if !input.iter().all(|v| v.is_finite()) { return Err("hybrid Delta finite input".into()); }
        let state = crate::prefix_hybrid_codec::decode_for_delta(packet)?;
        Ok(Self { request: r.clone(), input, state })
    }

    pub(crate) fn evaluate<F, B>(&self, r: &Request, m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
    where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer {
        self.check_identity(r)?;
        Self::run(r, &self.input, self.state.clone(), m, read)
    }

    /// A query consumes its decoded state once. Move it into recurrence rather
    /// than allocating and copying another 2 MiB working buffer.
    pub(crate) fn evaluate_owned<F, B>(self, r: &Request, m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
    where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer {
        self.check_identity(r)?;
        let Self { input, state, .. } = self;
        Self::run(r, &input, state, m, read)
    }

    fn check_identity(&self, r: &Request) -> Result<()> {
        let bound = &self.request;
        if r.version != bound.version || r.model != bound.model
            || r.pack_hash != bound.pack_hash || r.input_hash != bound.input_hash
            || r.step != bound.step || r.op != bound.op || r.tensor != bound.tensor
            || r.dims != bound.dims || r.aux != bound.aux || r.encoding != bound.encoding
            || r.scalars.len() != bound.scalars.len()
            || !r.scalars.iter().zip(&bound.scalars).all(|(a,b)| a.to_bits() == b.to_bits())
        { return Err("hybrid Delta request identity".into()); }
        Ok(())
    }

    fn run<F, B>(r: &Request, input: &[f32], state: crate::delta_full_log::InitialState, m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
    where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer {
        let mut inner = r.clone();
        inner.op = "delta_full_log_integer".into();
        inner.encoding = "bf16-block256-exact-v1".into();
        crate::delta_full_log::evaluate_from_state(&inner, input, m, read, Some(state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),
            "pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,
            "op":"delta_full_hybrid_integer","encoding":NAME,"dims":[1,32,2,0],
            "tensor":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight","scalars":[]})).unwrap()
    }
    #[cfg(feature="experimental-update-token-chunks")]
    #[test]
    fn recurrence_carry_keeps_outputs_and_final_state_across_token_boundary() {
        let n=90;let width=8;
        let make=|count:usize,salt:usize|->Vec<f32>{(0..count).map(|i|crate::bf(((i*17+salt)%31) as f32/64.-0.2)).collect()};
        let q=make(n*width,1);let k=make(n*width,2);let v=make(n*width,3);
        let g=vec![0.9375;n];let beta=vec![0.375;n];
        let initial=make(width*width,4);let mut whole=initial.clone();
        let expected=crate::delta(&q,&k,&v,&g,&beta,&mut whole,width,width).unwrap();
        let mut tiled=initial;let mut got=vec![];
        for (begin,end) in [(0,57),(57,n)] {
            got.extend(crate::delta(&q[begin*width..end*width],&k[begin*width..end*width],&v[begin*width..end*width],
                &g[begin..end],&beta[begin..end],&mut tiled,width,width).unwrap());
        }
        assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert_eq!(tiled.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),whole.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        #[cfg(feature="experimental-delta-state-layout")]
        {
            let value_major=make(width*width,4);
            let mut key_major:Vec<_>=(0..width*width).map(|i|value_major[(i%width)*width+i/width]).collect();
            let mut key_out=vec![];
            for (begin,end) in [(0,57),(57,n)] {
                key_out.extend(crate::delta_from_key_major(&q[begin*width..end*width],&k[begin*width..end*width],&v[begin*width..end*width],
                    &g[begin..end],&beta[begin..end],&mut key_major,width,width).unwrap());
            }
            assert_eq!(key_out.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            for i in 0..width*width {assert_eq!(key_major[i].to_bits(),whole[(i%width)*width+i/width].to_bits());}
        }
    }
    #[cfg(feature="experimental-update-token-chunks")]
    #[test]
    fn stream_rejects_stale_scope_and_cannot_restart_consumed_state() {
        let r=request();
        let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};
        let mut stream=ServerDeltaStream{bound:r.clone(),state:Some(crate::delta_full_log::InitialState::ValueMajor(vec![0.;32*128*128])),next:57};
        for field in 0..5 {
            let mut changed=r.clone();let mut offset=57;
            match field {0=>offset=0,1=>changed.tensor="wrong-layer".into(),2=>changed.input_hash="d".repeat(64),3=>changed.model="d".repeat(64),_=>changed.dims[2]+=1}
            assert!(stream.evaluate(&changed,&[],offset,&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid scope read")}).is_err());
            assert!(stream.state.is_some());
        }
        let mut oversized=r.clone();oversized.dims[0]=if cfg!(feature="experimental-adaptive-token-tiles"){90}else{58};
        assert!(stream.evaluate(&oversized,&[],57,&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid tile read")}).is_err());
        // A valid scope with missing weights consumes the carry and fails.
        let x=vec![0.;2560+3*8192];
        assert!(stream.evaluate(&r,&x,57,&m,&mut|_,_|->Result<Vec<u8>>{panic!("missing weight read")}).is_err());
        assert!(stream.state.is_none());
        assert_eq!(stream.evaluate(&r,&x,57,&m,&mut|_,_|->Result<Vec<u8>>{panic!("failed carry read")}).unwrap_err(),"server Delta stream identity/progress");
    }
    #[test]
    fn exact_packet_enters_typed_path_and_rejects_wrong_prefix() {
        let r = request();
        let log = vec![0.; 2 * 6176];
        let (packet, expected) = crate::prefix_hybrid_codec::prepare(&log, 2).unwrap();
        let count = 2560 + 3 * 8192;
        let mut payload = vec![1];
        payload.extend((count as u32).to_le_bytes());
        payload.extend(vec![0; count * 2]);
        payload.extend((packet.len() as u32).to_le_bytes());
        payload.extend(&packet);
        let input = PreparedDeltaHybrid::decode(&r, &payload).unwrap();
        assert!(input.state.value_major().iter().zip(expected).all(|(a,b)| a.to_bits() == b.to_bits()));
        let mut changed = r.clone(); changed.dims[2] = 3;
        assert!(PreparedDeltaHybrid::decode(&changed, &payload).is_err());
        assert!(PreparedDeltaHybrid::decode(&r, &payload[..payload.len()-1]).is_err());
        let mut changed = r.clone(); changed.dims[3] = 1;
        assert!(PreparedDeltaHybrid::decode(&changed, &payload).is_err());
        let mut changed = r.clone(); changed.step = 1;
        let m = Manifest { version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![] };
        assert!(input.evaluate(&changed, &m, &mut |_,_| -> Result<Vec<u8>> {panic!("identity failure must precede reads")}).is_err());
    }
}
