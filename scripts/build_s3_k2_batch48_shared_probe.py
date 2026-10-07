#!/usr/bin/env python3
"""Share the large normal body, with measured positive-zero initialization."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s3-k2-batch48-probe-v1/frozen-builder.py'
 d=ROOT/'artifacts/s3-k2-batch48-shared-probe-v1';d.mkdir(exist_ok=False)
 source=original.read_text().replace('s3-k2-batch48-probe-v1','s3-k2-batch48-shared-probe-v1')
 anchor="audit=json.loads((kernels/'layout-audit.json').read_text())"
 assert source.count(anchor)==1
 source=source.replace(anchor,"kr['kernels']=[k for k in kr['kernels'] if not k['seed']];"+anchor)
 before='let len=q.rows()*rows;let mut out=Vec::<core::mem::MaybeUninit<f32>>::with_capacity(len);out.set_len(len);'
 assert source.count(before)==1;source=source.replace(before,'let len=q.rows()*rows;let mut out=vec![0.0f32;len];')
 before='let f=if block==0{crate::s2_kernel::tile128_seed}else{crate::s2_kernel::tile128};'
 assert source.count(before)==1;source=source.replace(before,'let f=crate::s2_kernel::tile128;')
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
