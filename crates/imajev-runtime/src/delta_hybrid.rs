//! Exact client-held prefix packet consumed directly by the full Delta path.
use crate::{Manifest, Request, Result, prepared_weights::WeightBuffer};
pub(crate) const NAME: &str = "delta-hybrid-prefix-exact-v1";

/// Fields are constructed only after validating the envelope and exact packet.
pub struct PreparedDeltaHybrid {
    request: Request,
    input: Vec<f32>,
    state: crate::delta_full_log::InitialState,
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
