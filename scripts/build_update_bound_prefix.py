#!/usr/bin/env python3
"""Bind immutable server-held prefix state once, preserving packet decode results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    upstream=ROOT/'artifacts/update-f32-shapes-v1'
    hashes=json.loads((upstream/'workflow-hashes.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    d=ROOT/'artifacts/update-bound-prefix-v1'
    d.mkdir(exist_ok=False)
    for p in upstream.glob('*.wat'): (d/p.name).write_bytes(p.read_bytes())
    helper='''
/// Immutable fixed prefix decoded by the original checked server path.
/// No request activation or suffix state is retained here.
pub struct ServerDeltaPrefix { bound:Request, state:std::rc::Rc<Vec<f32>> }
impl ServerDeltaPrefix {
 pub(crate) fn new(input:PreparedDeltaHybrid)->Result<Self> {
  let PreparedDeltaHybrid{request,input:_,state}=input;
  let state=match state {crate::delta_full_log::InitialState::KeyMajor(v)=>v,
   _=>return Err("server prefix key-major layout".into())};
  if state.len()!=32*128*128 || !state.iter().all(|v|v.is_finite()) {return Err("server prefix finite state".into());}
  Ok(Self{bound:request,state:std::rc::Rc::new(state)})
 }
 pub(crate) fn input(&self,r:&Request,input:&[f32])->Result<PreparedDeltaHybrid> {
  let b=&self.bound;
  if r.version!=b.version || r.model!=b.model || r.pack_hash!=b.pack_hash || r.tensor!=b.tensor
   || r.op!=b.op || r.encoding!=b.encoding || r.aux!=b.aux || r.scalars.len()!=b.scalars.len()
   || !r.scalars.iter().zip(&b.scalars).all(|(x,y)|x.to_bits()==y.to_bits())
   || r.dims.len()!=4 || !(1..=90).contains(&r.dims[0]) || r.dims[1..]!=b.dims[1..]
   || input.len()!=r.dims[0]*2560+3*8192
   || !input.iter().all(|v|v.is_finite() && v.to_bits()&65535==0) {
   return Err("server prefix input identity/precision".into());
  }
  Ok(PreparedDeltaHybrid{request:r.clone(),input:input.to_vec(),
   state:crate::delta_full_log::InitialState::KeyMajor(self.state.as_ref().clone())})
 }
}
'''
    api='''
#[cfg(feature="experimental-prefix-hybrid")]
pub use delta_hybrid::ServerDeltaPrefix;
#[cfg(feature="experimental-prefix-hybrid")]
pub fn prepare_server_delta_prefix(r:&Request,input:&[f32],packet:&[u8])->Result<ServerDeltaPrefix> {
 let decoded=server_delta_hybrid_input(r,input,packet)?;
 match decoded {DecodedQueryInput::DeltaHybrid(v)=>ServerDeltaPrefix::new(v),
  _=>Err("server prefix input variant".into())}
}
#[cfg(feature="experimental-prefix-hybrid")]
pub fn server_delta_bound_input(r:&Request,input:&[f32],prefix:&ServerDeltaPrefix)->Result<DecodedQueryInput> {
 Ok(DecodedQueryInput::DeltaHybrid(prefix.input(r,input)?))
}
'''
    (d/'prefix-helper.rs').write_text(helper)
    (d/'prefix-api.rs').write_text(api)
    source=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-f32-shapes-v1','artifacts/update-bound-prefix-v1')
    anchor="    runtime = base['runtime_command'][:]"
    assert source.count(anchor)==1
    addition='''    p=D/'runtime/delta_hybrid.rs'
    p.write_text(p.read_text()+(D.parent/'prefix-helper.rs').read_text())
    p=D/'runtime/lib.rs'
    p.write_text(p.read_text()+(D.parent/'prefix-api.rs').read_text())
    p=D/'runtime/prefix_state_cache.rs'
    p.write_text(p.read_text()+'\\npub fn clear(){CACHE.with(|c|c.borrow_mut().clear());}\\n')
    p=D/'update_inference.rs'
    text=p.read_text()
    before='struct Prefix { values: Vec<f32>, packet: Vec<u8> }'
    assert text.count(before)==1
    text=text.replace(before,'struct Prefix { values: Vec<f32>, packet: Vec<u8>, prepared: Option<imajev_runtime::ServerDeltaPrefix> }')
    before='    if i%4==3 {\\n        if values.len()!=p*2048'
    assert text.count(before)==1
    text=text.replace(before,'    let mut prepared=None;\\n'+before)
    before='        imajev_runtime::server_delta_hybrid_input(&r,&input,&packet)?;'
    assert text.count(before)==1
    text=text.replace(before,'        if GRAPH.with(|g|g.borrow().prefix.is_empty()) {imajev_runtime::prefix_state_cache::clear();}\\n        prepared=Some(imajev_runtime::prepare_server_delta_prefix(&r,&input,&packet)?);')
    before='Prefix{values,packet:packet.into_vec()}'
    assert text.count(before)==1
    text=text.replace(before,'Prefix{values,packet:packet.into_vec(),prepared}')
    before='imajev_runtime::server_delta_hybrid_input(&r,&x,&p.packet).unwrap_or_else(|e|ic_cdk::trap(&e))'
    assert text.count(before)==1
    text=text.replace(before,'imajev_runtime::server_delta_bound_input(&r,&x,p.prepared.as_ref().expect("prepared fixed Delta prefix")).unwrap_or_else(|e|ic_cdk::trap(&e))')
    p.write_text(text)
'''
    source=source.replace(anchor,addition+anchor)
    (d/'frozen-builder.py').write_text(source)
    paths=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',d/'frozen-builder.py',d/'prefix-helper.rs',d/'prefix-api.rs']+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2)+'\n')
    exec(compile(source,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__': main()
