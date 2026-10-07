#!/usr/bin/env python3
"""Amortize fixed rank49 transforms across 88 outputs within the IC local limit."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'artifacts/s2-pair-factored-v1'
 r=json.loads((upstream/'build/report.json').read_text())
 for k in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in r[k].items())
 d=ROOT/'artifacts/s2-pair88-v1';d.mkdir(exist_ok=False)
 s=(upstream/'frozen-builder.py').read_text().replace('artifacts/s2-pair-factored-v1/build','artifacts/s2-pair88-v1/build')
 assert s.count('for j in range(8):')==2;s=s.replace('for j in range(8):','for j in range(11):')
 for before,after in [
  ("for j in range(16):\n        lines += [f'(local $sw", "for j in range(22):\n        lines += [f'(local $sw"),
  ("for j in range(16):\n        lines += [f'(local.set $sw", "for j in range(22):\n        lines += [f'(local.set $sw"),
  ('(i32.shl (i32.add (local.get $t) (i32.const {ti})) (i32.const 8))','(i32.mul (i32.add (local.get $t) (i32.const {ti})) (i32.const 352))'),
  ("replace('_n*32', '_n*64')", "replace('_n*32', '_n*88')")]:
  assert s.count(before)==1,(before,s.count(before));s=s.replace(before,after)
 anchor="    (src/'s2.rs').write_text(code)";assert s.count(anchor)==1
 addition='''    code=code.replace('rows%64!=0','rows%8!=0')
    before='  let stride=cols/4;let plane=(rows/4)*stride;let mut data=vec![0i8;w.len()];'
    assert code.count(before)==1
    code=code.replace(before,'  let original_rows=rows;let rows=rows.div_ceil(88)*88;let mut padded=vec![0i8;rows*cols];padded[..original_rows*cols].copy_from_slice(w);let w=&padded[..];\\n'+before)
    before='   for r in(0..rows).step_by(64){let wp:[*const i8;16]=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*stride+(r/4)*stride));let mut sums=vec![[0f32;64];q.rows()];'
    assert code.count(before)==1
    after='   for r in(0..rows).step_by(88){let width=(rows-r).min(88);let wp:[*const i8;16]=core::array::from_fn(|m|self.data.as_ptr().add(m*(self.rows/4)*stride+(r/4)*stride));let mut sums=vec![[0f32;88];q.rows()];let mut scales=[1f32;88];scales[..width].copy_from_slice(&sw[r..r+width]);'
    code=code.replace(before,after).replace('sw.as_ptr().add(r),sums','scales.as_ptr(),sums')
    before='out[t*rows+r..t*rows+r+64].copy_from_slice(&sums[t]);'
    assert code.count(before)==1
    code=code.replace(before,'out[t*rows+r..t*rows+r+width].copy_from_slice(&sums[t][..width]);')
'''
 s=s.replace(anchor,addition+anchor).replace('output_tile=64','output_tile=88')
 (d/'frozen-builder.py').write_text(s)
 (d/'upstream-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),upstream/'frozen-builder.py',upstream/'build/report.json']},indent=2)+'\n')
 exec(compile(s,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
