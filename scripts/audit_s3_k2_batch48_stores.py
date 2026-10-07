#!/usr/bin/env python3
"""Audit every emitted root reference, byte shuffle, scale and output store."""
from pathlib import Path
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s3-k2-batch48-kernels-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=json.loads((D/'report.json').read_text());ns={'__name__':'plan','__file__':str(D/'plan.py')};exec(compile((D/'plan.py').read_text(),str(D/'plan.py'),'exec'),ns)
 _,_,c,_,roots,_=ns['plan']();_,regs,_=ns['reconstruct'](c,roots);checked=0
 for item in report['kernels']:
  text=(ROOT/item['path']).read_text();fragments=text.split('f32x4.convert_i32x4_s');assert len(fragments)==1537
  index=0
  for ti in range(6):
   for j in range(4):
    for row in range(8):
     for lane in range(4):
      for base in [0,4]:
       before=fragments[index];after=fragments[index+1].split('v128.store',1)[0]
       references=re.findall(r'local.get \$(p\d+_\d+_\d+|pc\d+)',before)[-4:]
       expected=[f'p{ti}_{j}_{regs[row][base+k]}'if regs[row][base+k]<343 else f'pc{regs[row][base+k]}'for k in range(4)]
       assert references==expected
       masks=re.findall(r'i8x16.shuffle ((?:\d+ ){15}\d+)',before)[-3:];assert len(masks)==3
       # Tag each root/lane independently; the actual byte masks must produce
       # contiguous physical rows lane*8+base..+3 in this tile32 group.
       inputs=[np.array([ni+ln*8 for ln in range(4)],dtype=np.int32)for ni in range(base,base+4)]
       def shuffle(x,y,mask):return np.concatenate([x.view(np.uint8),y.view(np.uint8)])[list(map(int,mask.split()))].copy().view(np.int32)
       got=shuffle(shuffle(inputs[0],inputs[1],masks[0]),shuffle(inputs[2],inputs[3],masks[1]),masks[2])
       assert np.array_equal(got,np.arange(lane*8+base,lane*8+base+4))
       assert re.findall(r'local.get \$sx(\d+)',after)==[str(ti*8+row)]
       assert re.findall(r'local.get \$sw(\d+)',after)==[str(j*8+lane*2+base//4)]
       assert int(re.search(r'v128.store offset=(\d+)',fragments[index+1])[1])==j*128+(lane*8+base)*4
       index+=1;checked+=1
 files=[Path(__file__),D/'report.json',D/'plan.py']+[ROOT/k['path']for k in report['kernels']]
 r=dict(complete=True,vector_stores_checked=checked,all_emitted_root_masks_scales_offsets_exact=True,
  wasm_execution_verified=False,performance_verified=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (D/'store-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(complete=True,vector_stores_checked=checked)))
if __name__=='__main__':main()
