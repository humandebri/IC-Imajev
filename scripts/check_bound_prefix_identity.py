#!/usr/bin/env python3
"""Run the actual opaque-prefix helper in a native identity/alias boundary harness."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/update-bound-prefix-v1/identity'
    d.mkdir(exist_ok=False)
    helper=d.parent/'prefix-helper.rs'
    source='''#![allow(dead_code)]
type Result<T>=std::result::Result<T,String>;
#[derive(Clone)]struct Request{version:u32,model:String,pack_hash:String,tensor:String,op:String,encoding:String,
 aux:Vec<String>,scalars:Vec<f32>,dims:Vec<usize>}
mod delta_full_log {pub enum InitialState{KeyMajor(Vec<f32>),ValueMajor(Vec<f32>)}}
struct PreparedDeltaHybrid{request:Request,input:Vec<f32>,state:delta_full_log::InitialState}
include!(r"HELPER_PATH");
fn request()->Request{Request{version:1,model:"model".into(),pack_hash:"pack".into(),tensor:"layer0.qkv".into(),
 op:"delta_full_hybrid_integer".into(),encoding:"delta-hybrid-prefix-exact-v1".into(),aux:vec![],scalars:vec![],dims:vec![1,32,38,0]}}
fn make(state:Vec<f32>)->PreparedDeltaHybrid{PreparedDeltaHybrid{request:request(),input:vec![0.;2560+3*8192],state:delta_full_log::InitialState::KeyMajor(state)}}
fn main(){
 let values:Vec<f32>=(0..32*128*128).map(|i|f32::from_bits([0,0x80000000,1,0x80000001,0x3f123456,0xbf654321][i%6])).collect();
 let prefix=ServerDeltaPrefix::new(make(values.clone())).unwrap();
 for n in [1,48,56,57,90] {
  let mut r=request();r.dims[0]=n;
  let input:Vec<f32>=(0..n*2560+3*8192).map(|i|f32::from_bits([0,0x80000000,0x00010000,0x80010000,0x3f120000,0xbf650000][i%6])).collect();
  let actual=prefix.input(&r,&input).unwrap();
  assert!(actual.input.iter().zip(&input).all(|(a,b)|a.to_bits()==b.to_bits()));
  match actual.state {delta_full_log::InitialState::KeyMajor(mut s)=>{
   assert!(s.iter().zip(&values).all(|(a,b)|a.to_bits()==b.to_bits()));s.fill(9.);
  },_=>panic!("layout")}
  assert!(prefix.state.iter().zip(&values).all(|(a,b)|a.to_bits()==b.to_bits()));
 }
 let input=vec![0.;2560+3*8192];
 let mut invalid=vec![];
 for field in 0..9 {let mut r=request();match field {
  0=>r.version+=1,1=>r.model="other".into(),2=>r.pack_hash="other".into(),3=>r.tensor="other".into(),
  4=>r.op="other".into(),5=>r.encoding="other".into(),6=>r.aux.push("other".into()),
  7=>r.scalars.push(0.),8=>r.dims.clear(),_=>unreachable!()};invalid.push(r);}
 for dims in [vec![0,32,38,0],vec![91,32,38,0],vec![1,31,38,0],vec![1,32,27,0],vec![1,32,38,1],vec![1,32,38]] {
  let mut r=request();r.dims=dims;invalid.push(r);
 }
 for r in &invalid {assert!(prefix.input(r,&input).is_err());}
 assert!(prefix.input(&request(),&input[..input.len()-1]).is_err());
 let mut longer=input.clone();longer.push(0.);assert!(prefix.input(&request(),&longer).is_err());
 for bits in [1,0x3f800001,0x7f800000,0xff800000,0x7fc10000] {
  let mut x=input.clone();x[0]=f32::from_bits(bits);assert!(prefix.input(&request(),&x).is_err());
 }
 assert!(ServerDeltaPrefix::new(make(vec![0.;32*128*128-1])).is_err());
 let mut bad=values.clone();bad[0]=f32::INFINITY;assert!(ServerDeltaPrefix::new(make(bad)).is_err());
 let mut bad=make(values);bad.state=delta_full_log::InitialState::ValueMajor(vec![]);assert!(ServerDeltaPrefix::new(bad).is_err());
 println!("5 valid token counts; 15 request mutations; 7 input boundaries; 3 state boundaries; independent state copies verified");
}
'''.replace('HELPER_PATH',str(helper))
    (d/'harness.rs').write_text(source)
    command=['rustc','--edition=2021',str(d/'harness.rs'),'-o',str(d/'harness')]
    with (d/'compiler.log').open('w') as log: subprocess.run(command,check=True,stdout=log,stderr=log)
    output=subprocess.check_output([str(d/'harness')],text=True)
    result=dict(passed=True,helper_sha256=sha(helper),source_sha256=sha(Path(__file__)),harness_sha256=sha(d/'harness.rs'),command=command,output=output,
                scope='Actual helper identity and immutable-state-copy behavior in a native boundary harness. Packet decode and Wasm full inference are separate proofs.')
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
