//! Correctness-only capture. Never included in performance candidates.
use std::cell::RefCell;
struct Capture {n:usize, bytes:Vec<u8>}
thread_local! {static CAPTURE:RefCell<Option<Capture>>=const{RefCell::new(None)};}
fn put(bytes:&mut[u8],offset:usize,values:&[f32]) {
 for (dst,value) in bytes[offset..offset+values.len()*4].chunks_exact_mut(4).zip(values) {dst.copy_from_slice(&value.to_le_bytes());}
}
pub(crate) fn begin(initial:&[f32],n:usize,key_major:bool,step:u64) {
 assert_eq!(initial.len(),32*16384);
 // Header, initial state, final state, then head-major Q/K/V/G/B/output.
 let mut bytes=vec![0u8;24+4*(2*32*16384+4*32*n*128+2*32*n)];
 bytes[..4].copy_from_slice(b"DLC1");
 bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
 bytes[8..12].copy_from_slice(&u32::from(key_major).to_le_bytes());
 bytes[12..16].copy_from_slice(&32u32.to_le_bytes());
 bytes[16..24].copy_from_slice(&step.to_le_bytes());
 put(&mut bytes,24,initial);
 CAPTURE.with(|c|*c.borrow_mut()=Some(Capture{n,bytes}));
}
pub(crate) fn head(h:usize,q:&[f32],k:&[f32],v:&[f32],g:&[f32],b:&[f32],state:&[f32],output:&[f32]) {
 CAPTURE.with(|c|{let mut c=c.borrow_mut();let c=c.as_mut().expect("capture begun");let n=c.n;
 assert!(h<32);assert_eq!(state.len(),16384);
 let state_size=32*16384;let vector_size=32*n*128;let scalar_size=32*n;
 put(&mut c.bytes,24+4*(state_size+h*16384),state);
 let base=24+8*state_size;
 for (i,x) in [q,k,v].iter().enumerate(){assert_eq!(x.len(),n*128);put(&mut c.bytes,base+4*(i*vector_size+h*n*128),x);}
 for (i,x) in [g,b].iter().enumerate(){assert_eq!(x.len(),n);put(&mut c.bytes,base+4*(3*vector_size+i*scalar_size+h*n),x);}
 assert_eq!(output.len(),n*128);
 put(&mut c.bytes,base+4*(3*vector_size+2*scalar_size+h*n*128),output);
 });
}
pub fn chunk(offset:u32,length:u32)->Result<Vec<u8>,String> {
 if length>524288{return Err("capture chunk limit".into());}
 CAPTURE.with(|c|{let c=c.borrow();let c=c.as_ref().ok_or("no capture")?;
 let lo=offset as usize;let hi=lo.checked_add(length as usize).ok_or("capture range")?;
 Ok(c.bytes.get(lo..hi).ok_or("capture range")?.to_vec())})
}
