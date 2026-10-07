#!/usr/bin/env python3
"""Compact initialized query coefficients and match current raw-project finite-check boundary."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s1-k2-direct-v1';d=ROOT/'artifacts/s1-k2-compact-v1';d.mkdir(exist_ok=False)
 s=(old/'frozen-builder.py').read_text().replace('s1-k2-direct-v1','s1-k2-compact-v1')
 patch='''
 p=src/'winograd.rs';text=p.read_text();edits=[
 ('let len=7*pairs*cols;','let len=7*pairs*(cols/2);'),
 ('let mut data=vec![0;len];unsafe{inputs(q,pairs,data.as_mut_ptr());data.set_len(len);}','let mut data=Vec::with_capacity(len);unsafe{inputs(q,pairs,data.as_mut_ptr());data.set_len(len);}'),
 ('m*a.pairs*self.cols+pair*self.cols+block*256+k','m*a.pairs*(self.cols/2)+pair*(self.cols/2)+block*128+k'),
 ('m*a.pairs*cols','m*a.pairs*(cols/2)'),
 ('m*pairs*cols+pair*cols+block*256+k','m*pairs*(cols/2)+pair*(cols/2)+block*128+k')]
 for before,after in edits:assert text.count(before)>0,before;text=text.replace(before,after)
 for m in range(7):
  before=f'out.add({m}*pairs*cols+pair*cols+block*256+k)'
  after=f'out.add({m}*pairs*(cols/2)+pair*(cols/2)+block*128+k)'
  assert text.count(before)==1;text=text.replace(before,after)
 before='if out.iter().any(|v|!v.is_finite()){return Err("wide finite output".into());}Ok(out)'
 assert text.count(before)==1;text=text.replace(before,'Ok(out)')
 p.write_text(text);(d/'compact-edits.json').write_text(json.dumps(edits,indent=2)+'\\n')
 for path,_ in generated:
  text=path.read_text()
  before='(local.set $qo(i32.shl(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start))(i32.const 1)))'
  after='(local.set $qo(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start)))'
  assert text.count(before)==3;text=text.replace(before,after);path.write_text(text)
'''
 anchor=" p=src/'lib.rs'\n cmd=nb['command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 (d/'frozen-builder.py').write_text(s);(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),old/'build/report.json',old/'frozen-builder.py',d/'frozen-builder.py']},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
