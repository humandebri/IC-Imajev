#!/usr/bin/env python3
"""Add independent preactivation checks and one unambiguous runtime dependency."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/build_conv4_probe.py';s=p.read_text().replace('artifacts/conv4-simd-v1','artifacts/conv4-simd-v2')
 s=s.replace('pub unsafe fn conv4_simd(', 'pub unsafe fn conv4_simd<const ACT:bool>(')
 s=s.replace(' for ch in (0..c).step_by(4){',' let emit=|v|if ACT{bf_silu(bf(v))}else{v};\n for ch in (0..c).step_by(4){')
 for i in range(4):s=s.replace(f'bf_silu(bf(f32x4_extract_lane::<{i}>(sum)))',f'emit(f32x4_extract_lane::<{i}>(sum))')
 s=s.replace('assert!(method<=1&&input.len()', 'assert!(method<=3&&input.len()')
 before='let out=if method==0 {imajev_runtime::execute(&r,x,w).unwrap()}else{unsafe{imajev_runtime::conv4_simd(x,w,n,c)}};'
 assert s.count(before)==1
 after='''let out=match method {0=>imajev_runtime::execute(&r,x,w).unwrap(),1=>unsafe{imajev_runtime::conv4_simd::<true>(x,w,n,c)},2=>{let mut y=vec![0.;n*c];for t in 0..n{for ch in 0..c{let mut sum=0.;for j in 0..4{sum+=x[(t+j)*c+ch]*w[ch*4+j];}y[t*c+ch]=sum;}}y.extend_from_slice(&x[x.len()-3*c..]);y},_=>unsafe{imajev_runtime::conv4_simd::<false>(x,w,n,c)}};'''
 s=s.replace(before,after)
 before=" command+=['--extern','imajev_runtime='+str(d/'libimajev_runtime.rlib')]"
 assert s.count(before)==1
 after=""" command=[v if not v.startswith('imajev_runtime=') else 'imajev_runtime='+str(d/'libimajev_runtime.rlib') for v in command]
 command+=['--extern',next(cmd[i+1] for i,v in enumerate(cmd) if v=='--extern' and cmd[i+1].startswith('serde_json='))]
 for i,v in enumerate(cmd):
  if v=='-L':command+=['-L',cmd[i+1]]
"""
 s=s.replace(before,after)
 d=ROOT/'artifacts/conv4-simd-v2';d.mkdir(exist_ok=False);(d/'frozen-builder.py').write_text(s)
 (d/'builder-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest(),str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
