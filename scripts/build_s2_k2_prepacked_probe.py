#!/usr/bin/env python3
"""Prototype immutable rank49 coefficient preparation; inference reads only precomputed values."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/build_s2_k2_probe.py';d=ROOT/'artifacts/s2-k2-prepacked-probe-v1';d.mkdir(exist_ok=False)
 s=old.read_text().replace('s2-k2-probe-v1','s2-k2-prepacked-probe-v1').replace('s2-k2-kernels-v1','s2-k2-prepacked-kernels-v1').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 patch=''' p=src/'s2.rs';text=p.read_text().replace('data:Vec<i8>','data:Vec<i16>')
 lo=text.index('  let stride=cols/4;let plane=');hi=text.index('  Ok(Self',lo)
 text=text[:lo]+"  let blocks=cols/256;let mut data=vec![0i16;(rows/16)*blocks*49*256];\\n  for group in 0..rows/16{for block in 0..blocks{for m in 0..49{for k in 0..64{for lane in 0..4{let value:i16=coeff::B[m].iter().map(|&(j,c)|c*w[(group*16+lane*4+j%4)*cols+block*256+j/4*64+k]as i16).sum();data[((group*blocks+block)*49+m)*256+(k/2)*8+lane*2+k%2]=value;}}}}}\\n"+text[hi:]
 text=text.replace('pub fn bytes(&self)->usize{self.data.len()}','pub fn bytes(&self)->usize{self.data.len()*2}')
 before='let wp:[*const i8;16]=core::array::from_fn(|m|self.data.as_ptr().add(m*plane+(r/16)*cols));'
 after='let wp=[self.data.as_ptr().add((r/16)*blocks*49*256)];'
 assert text.count(before)==1;text=text.replace(before,after)
 before='let value:i16=coeff::B[m].iter().map(|&(j,c)|c*self.data[j*(self.rows*cols/16)+(row/4)*cols+block*256+(k/2)*8+(row%4)*2+k%2]as i16).sum();'
 after='let value=self.data[(((row/4)*blocks+block)*49+m)*256+(k/2)*8+(row%4)*2+k%2];'
 assert text.count(before)==1;text=text.replace(before,after);p.write_text(text)
'''
 anchor=" cmd=nb['command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 (d/'frozen-builder.py').write_text(s);files=[Path(__file__),old,d/'frozen-builder.py'];(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
