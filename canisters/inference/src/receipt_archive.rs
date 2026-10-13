//! Bounded stable receipts. Only keys/locations stay on the Wasm heap.
//! The model is immutable once ready; its upload range precedes this archive.
use super::*;
use super::paid_types::{Receipt, ReceiptState, Refund};
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

pub(super) const CAPACITY: usize = 4096;
const PAGE: u64 = 65536;
const SLOT: u64 = PAGE;
const HEADER: usize = 64;
// Upgrade metadata is bounded to <2 MB. Keep its region separate from receipts.
const METADATA_RESERVE: u64 = 4 * 1024 * 1024;
#[derive(Default, Serialize, Deserialize)]
pub(super) struct Archive {
    base: Option<u64>,
    slots: u32,
    #[serde(skip)]
    entries: BTreeMap<[u8;32], Entry>,
    #[serde(skip)]
    free: Vec<u32>,
}
struct Entry {slot:u32, bytes:u32, job_id:u64, time:u64, refund:Refund}
fn key(caller:Principal, request_id:&str)->[u8;32] {
    let mut h=Sha256::new();h.update([caller.as_slice().len() as u8]);
    h.update(caller.as_slice());h.update(request_id.as_bytes());h.finalize().into()
}
fn refund_byte(r:&Refund)->u8 {match r {Refund::None=>0,Refund::Pending=>1,Refund::InFlight=>2,Refund::Done=>3}}
fn decode_refund(v:u8)->Refund {match v {0=>Refund::None,1=>Refund::Pending,2=>Refund::InFlight,3=>Refund::Done,_=>panic!("receipt archive refund")}}
#[cfg(not(test))]
fn read(at:u64, bytes:&mut[u8]) {ic_cdk::api::stable_read(at,bytes)}
#[cfg(not(test))]
fn write(at:u64, bytes:&[u8]) {ic_cdk::api::stable_write(at,bytes)}
#[cfg(not(test))]
fn grow(end:u64) {
    let pages=end.div_ceil(PAGE);let old=ic_cdk::api::stable_size();
    if pages>old {assert_ne!(ic_cdk::api::stable_grow(pages-old),u64::MAX,"receipt archive stable grow");}
}
#[cfg(test)]
thread_local! {static MEMORY:RefCell<Vec<u8>>=RefCell::new(vec![]);}
#[cfg(test)]
fn read(at:u64, bytes:&mut[u8]) {MEMORY.with(|m|bytes.copy_from_slice(&m.borrow()[at as usize..at as usize+bytes.len()]));}
#[cfg(test)]
fn write(at:u64, bytes:&[u8]) {MEMORY.with(|m|m.borrow_mut()[at as usize..at as usize+bytes.len()].copy_from_slice(bytes));}
#[cfg(test)]
fn grow(end:u64) {MEMORY.with(|m|{let mut m=m.borrow_mut();let size=end.div_ceil(PAGE) as usize*PAGE as usize;if size>m.len(){m.resize(size,0);}});}
#[cfg(test)]
fn base()->u64 {PAGE}
#[cfg(not(test))]
fn base()->u64 {STORE.with(|s|{let s=s.borrow();let m=s.manifest.as_ref().expect("receipt model");assert!(s.ready,"receipt model ready");m.bytes.div_ceil(PAGE)*PAGE+METADATA_RESERVE})}
impl Archive {
    pub fn len(&self)->usize {self.entries.len()}
    pub fn in_flight(&self)->bool {self.entries.values().any(|e|e.refund==Refund::InFlight)}
    fn at(&self, slot:u32)->u64 {self.base.expect("receipt archive base")+slot as u64*SLOT}
    fn load(&self, k:&[u8;32], e:&Entry)->Receipt {
        let mut bytes=vec![0;e.bytes as usize];read(self.at(e.slot)+HEADER as u64,&mut bytes);
        let row:Receipt=serde_json::from_slice(&bytes).expect("receipt archive record");
        assert_eq!(key(row.caller,&row.request_id),*k,"receipt archive identity");
        assert_eq!(row.job_id,e.job_id,"receipt archive job");row
    }
    pub fn find(&self, caller:Principal, request_id:&str)->Option<Receipt> {
        let k=key(caller,request_id);self.entries.get(&k).map(|e|self.load(&k,e))
    }
    pub fn by_job(&self, id:u64)->Option<Receipt> {
        self.entries.iter().find(|(_,e)|e.job_id==id).map(|(k,e)|self.load(k,e))
    }
    pub fn store(&mut self,row:&Receipt) {
        assert!(!matches!(row.state,ReceiptState::Running),"archive running receipt");
        let k=key(row.caller,&row.request_id);
        let bytes=serde_json::to_vec(row).expect("receipt archive encode");
        assert!(bytes.len()<=SLOT as usize-HEADER,"receipt archive record size");
        let slot=if let Some(e)=self.entries.get(&k) {assert_eq!(e.job_id,row.job_id,"receipt archive collision");e.slot}
            else if let Some(slot)=self.free.pop(){slot}
            else {assert!((self.slots as usize)<CAPACITY,"receipt archive capacity");let slot=self.slots;self.slots+=1;slot};
        if self.base.is_none(){self.base=Some(base());}
        // Keep an extra page for the upgrade metadata footer at stable_size-8.
        grow(self.at(slot)+SLOT+PAGE);
        let mut header=[0;HEADER];header[..4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        header[4..36].copy_from_slice(&k);header[36..44].copy_from_slice(&row.job_id.to_le_bytes());
        header[44..52].copy_from_slice(&row.time.to_le_bytes());header[52]=refund_byte(&row.refund);
        write(self.at(slot)+HEADER as u64,&bytes);write(self.at(slot),&header);
        self.entries.insert(k,Entry{slot,bytes:bytes.len() as u32,job_id:row.job_id,time:row.time,refund:row.refund.clone()});
    }
    pub fn expire(&mut self, now:u64, ttl:u64) {
        let expired:Vec<_>=self.entries.iter().filter(|(_,e)|matches!(e.refund,Refund::None|Refund::Done) && now.saturating_sub(e.time)>=ttl).map(|(k,_)|*k).collect();
        for k in expired {let e=self.entries.remove(&k).unwrap();write(self.at(e.slot),&0u32.to_le_bytes());self.free.push(e.slot);}
    }
    pub fn restore(&mut self) {
        self.entries.clear();self.free.clear();assert!((self.slots as usize)<=CAPACITY,"receipt archive slots");
        for slot in 0..self.slots {
            let mut h=[0;HEADER];read(self.at(slot),&mut h);let bytes=u32::from_le_bytes(h[..4].try_into().unwrap());
            if bytes==0 {self.free.push(slot);continue;}
            assert!(bytes as usize<=SLOT as usize-HEADER,"receipt archive header size");
            let k=h[4..36].try_into().unwrap();let e=Entry{slot,bytes,job_id:u64::from_le_bytes(h[36..44].try_into().unwrap()),time:u64::from_le_bytes(h[44..52].try_into().unwrap()),refund:decode_refund(h[52])};
            assert!(self.entries.insert(k,e).is_none(),"duplicate receipt archive key");
        }
    }
}
