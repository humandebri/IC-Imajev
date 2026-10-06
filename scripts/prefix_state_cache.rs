//! Owner-prepared immutable prefix states; exact-content lookups only.
use crate::Result;
use std::{cell::RefCell,rc::Rc};
const TOKENS:usize=27;
const VALUES:usize=TOKENS*6176;
const STATE:usize=32*128*128;
const MAX_ENTRIES:usize=24;
struct Entry {log:[u8;32],packet:[u8;32],state:Rc<Vec<f32>>}
thread_local! {static CACHE:RefCell<Vec<Entry>>=const{RefCell::new(Vec::new())};}
fn digest(bytes:&[u8])->[u8;32]{*blake3::hash(bytes).as_bytes()}
fn float_bytes(values:&[f32])->&[u8]{
 // Wasm F32 has initialized four-byte IEEE representation; this view only
 // hashes bytes of the live slice and never changes them.
 unsafe{std::slice::from_raw_parts(values.as_ptr().cast(),values.len()*4)}
}
pub(crate) fn lookup_log(n:usize,h:usize,values:&[f32])->Option<Vec<f32>> {
 if n!=TOKENS||h!=32||values.len()!=VALUES||CACHE.with(|c|c.borrow().is_empty()){return None;}
 let key=digest(float_bytes(values));CACHE.with(|c|c.borrow().iter().find(|e|e.log==key).map(|e|e.state.as_ref().clone()))
}
pub(crate) fn lookup_packet(bytes:&[u8])->Option<Vec<f32>> {
 if bytes.len()>1_990_000||CACHE.with(|c|c.borrow().is_empty()){return None;}
 let key=digest(bytes);CACHE.with(|c|c.borrow().iter().find(|e|e.packet==key).map(|e|e.state.as_ref().clone()))
}
/// This function must only be reachable through the canister's owner update.
/// Both representations are evaluated and compared before the entry is stored.
pub fn prepare(log_bytes:&[u8],packet:&[u8])->Result<(u32,u64)> {
 if log_bytes.len()!=VALUES*4||log_bytes.len()+packet.len()>1_990_000||packet.len()<12||&packet[4..8]!=(TOKENS as u32).to_le_bytes(){return Err("fixed prefix shape/size".into());}
 let lk=digest(log_bytes);let pk=digest(packet);
 if let Some(stats)=CACHE.with(|c|{let c=c.borrow();c.iter().any(|e|e.log==lk&&e.packet==pk).then_some((c.len()as u32,(c.len()*STATE*4)as u64))}){return Ok(stats);}
 if CACHE.with(|c|{let c=c.borrow();c.len()>=MAX_ENTRIES||c.iter().any(|e|e.log==lk||e.packet==pk)}){return Err("fixed prefix capacity/alias".into());}
 let values:Vec<_>=log_bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
 let state=crate::delta_log::restore_key_major(TOKENS,32,&values)?;
 let decoded=crate::prefix_hybrid_codec::decode_for_delta(packet)?;
 let actual=match decoded{crate::delta_full_log::InitialState::KeyMajor(v)=>v, _=>return Err("fixed prefix layout".into())};
 if state.len()!=STATE||actual.len()!=STATE||!state.iter().zip(&actual).all(|(a,b)|a.to_bits()==b.to_bits()){return Err("fixed prefix state mismatch".into());}
 CACHE.with(|c|{let mut c=c.borrow_mut();c.push(Entry{log:lk,packet:pk,state:Rc::new(state)});Ok((c.len()as u32,(c.len()*STATE*4)as u64))})
}
