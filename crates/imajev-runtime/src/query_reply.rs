//! Server-produced reply payloads; callers cannot construct an opaque payload.
use crate::{Request,Result};
/// Numeric replies retain the checked public encoder. Opaque replies are built
/// only by the runtime from validated/generated typed carry.
pub enum EvaluatedReply { Values(Vec<f32>), Payload(PayloadReply) }
/// ```compile_fail
/// let reply = imajev_runtime::PayloadReply { request: todo!(), payload: vec![] };
/// ```
pub struct PayloadReply { request:Request, payload:Vec<u8> }
impl PayloadReply {pub(crate) fn into_payload(self)->Vec<u8> {self.payload}}
impl EvaluatedReply {
    pub(crate) fn payload(r:&Request,payload:Vec<u8>)->Self {
        Self::Payload(PayloadReply{request:r.clone(),payload})
    }
    pub fn into_values(self)->Result<Vec<f32>> {
        match self {Self::Values(v)=>Ok(v),Self::Payload(_)=>Err("opaque carry is not a numeric terminal output".into())}
    }
    /// signed_transport requires authenticated owner ingress, exactly as the
    /// existing encode_signed_reply API. Only version3 omits its checksum.
    pub fn encode(self,r:&Request,signed_transport:bool)->Result<Vec<u8>> {
        let Self::Payload(p)=self else {
            let Self::Values(v)=self else {unreachable!()};
            #[cfg(feature="experimental-host-checksum")]
            if signed_transport {return crate::encode_signed_reply(r,&v);}
            return crate::encode(r,&v);
        };
        let mut expected=p.request;
        expected.step=expected.step.checked_add(1).ok_or("progress overflow")?;
        if !crate::same_request(r,&expected){return Err("opaque reply identity/progress".into());}
        let header=serde_json::to_vec(r).map_err(|e|e.to_string())?;
        if header.len()>16384 || 4+header.len()+p.payload.len()+32>2_000_000{return Err("opaque reply frame size".into());}
        let mut b=Vec::with_capacity(4+header.len()+p.payload.len()+32);
        b.extend_from_slice(&(header.len()as u32).to_le_bytes());b.extend(header);b.extend(p.payload);
        let digest=if signed_transport && cfg!(feature="experimental-host-checksum") && r.version==3 {[0;32]}else{crate::frame_digest(r.version,&b,true)?};
        b.extend(digest);Ok(b)
    }
}
