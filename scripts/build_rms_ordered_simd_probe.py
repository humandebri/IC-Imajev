#!/usr/bin/env python3
"""Measure exact F32 RMS SIMD products while preserving scalar sum order."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/rms-ordered-simd-v1';d.mkdir(exist_ok=False)
 helper=r'''#[target_feature(enable="simd128")]
unsafe fn rms_ordered_simd(x:&[f32],w:&[f32],eps:f32)->Vec<f32>{
use core::arch::wasm32::*;
let c=x.len();let mut total=0f32;let mut j=0;
while j+4<=c{let v=v128_load(x.as_ptr().add(j).cast());let squares=f32x4_mul(v,v);
total+=f32x4_extract_lane::<0>(squares);total+=f32x4_extract_lane::<1>(squares);total+=f32x4_extract_lane::<2>(squares);total+=f32x4_extract_lane::<3>(squares);j+=4;}
while j<c{let v=x[j];total+=v*v;j+=1;}
let scale=(total/c as f32+eps).sqrt().recip();let sv=f32x4_splat(scale);let mut out=vec![0.;c];j=0;
while j+4<=c{let v=v128_load(x.as_ptr().add(j).cast());let weight=v128_load(w.as_ptr().add(j).cast());v128_store(out.as_mut_ptr().add(j).cast(),f32x4_mul(f32x4_mul(v,sv),weight));j+=4;}
while j<c{out[j]=x[j]*scale*w[j];j+=1;}out
}
'''
 p=ROOT/'scripts/build_add_norm_simd_probe.py';s=p.read_text().replace('artifacts/add-norm-simd-v1/build','artifacts/rms-ordered-simd-v1/build').replace('imajev_add_norm_simd_probe','imajev_rms_ordered_simd_probe')
 lo=s.index(" helper=r'''");hi=s.index(" (d/'add_norm_simd.rs').write_text(helper)",lo);s=s[:lo]+' helper='+repr(helper)+'\n'+s[hi:];s=s.replace('add_norm_simd.rs','rms_ordered_simd.rs')
 lo=s.index('let y=if method==0');hi=s.index('\nlet end=',lo)
 s=s[:lo]+'''let mut y=Vec::with_capacity(n*c);for row in x[..n*c].chunks_exact(c){if method==0{y.extend(rms(row,w,eps));}else{y.extend(unsafe{rms_ordered_simd(row,w,eps)});}};'''+s[hi:]
 s=s.replace("Original scalar add_norm expression and rms body copied verbatim; SIMD component products preserve scalar lane sum and F32 multiply order.","Original rms body copied verbatim; SIMD products preserve scalar lane sum and both separate F32 multiplies, without BF16 rounding.")
 (d/'frozen-builder.py').write_text(s);(d/'entry-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v) for v in [Path(__file__),p,d/'frozen-builder.py']},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
