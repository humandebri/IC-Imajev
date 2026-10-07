#!/usr/bin/env python3
"""Hold the immutable activation-table borrow once per convolution, not per scalar."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'artifacts/conv4-simd-v3/frozen-builder.py';s=p.read_text().replace('artifacts/conv4-simd-v3','artifacts/conv4-cached-v1')
 anchor=" shutil.copytree(base/'runtime',d/'runtime')";assert s.count(anchor)==1
 api='''
pub fn with_table<const K:usize,R>(run:impl FnOnce(Option<&[f32;N]>)->R)->R{
 const{assert!(K<4);}
 FIXED.with(|t|{let t=t.borrow();run(t.as_ref().map(|t|&*t.values[K]))})
}
'''
 s=s.replace(anchor,anchor+"\n p=d/'runtime/prepared_activation.rs';p.write_text(p.read_text()+"+repr(api)+")")
 before=' let emit=|v|if ACT{bf_silu(bf(v))}else{v};'
 assert s.count(before)==1
 after=''' let run=|table:Option<&[f32;65536]>|{
 let emit=|v|if ACT{
  let b=bf(v);if b.is_finite(){if let Some(t)=table{return t[(b.to_bits()>>16)as usize];}}
  bf_silu_original(b)
 }else{v};'''
 s=s.replace(before,after)
 before=' y.extend_from_slice(&x[x.len()-3*c..]);y\n}'
 assert s.count(before)==1
 s=s.replace(before,' y.extend_from_slice(&x[x.len()-3*c..]);y\n };\n if ACT{prepared_activation::with_table::<1,_>(run)}else{run(None)}\n}')
 d=ROOT/'artifacts/conv4-cached-v1';d.mkdir(exist_ok=False);(d/'frozen-builder.py').write_text(s)
 files=[p,Path(__file__),ROOT/'artifacts/conv4-simd-v3/builder-hashes.json']
 (d/'builder-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
