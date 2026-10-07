#!/usr/bin/env python3
"""One four-plane raw payload can address both rank7 and contiguous rank49 roots."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/contiguous-common-raw-layout-v1';d.mkdir(exist_ok=False);cases=[];rng=np.random.default_rng(7049)
 for rows in [16,32,96,256]:
  for cols in [256,512,2560,9216]:
   rr,cc=np.indices((rows,cols));plane=rows*cols//4
   index=((cc%256)//128*2+(rr%8)//4)*plane+(rr//8)*cols*2+(cc//256)*512+((cc%128)//2)*8+(rr%4)*2+cc%2
   assert np.array_equal(np.sort(index.ravel()),np.arange(rows*cols));origins=np.empty(rows*cols,dtype=np.int64);origins[index]=(rr*cols+cc)
   group,block,k,lane,sub=np.indices((rows//16,cols//256,32,4,2))
   for ki in range(4):
    for ni in range(4):
     pointer=((ki//2)*2+ni%2)*plane+(group*2+ni//2)*cols*2+(ki%2)*256
     address=pointer+block*512+k*8+lane*2+sub
     expected=(group*16+ni*4+lane)*cols+block*256+ki*64+k*2+sub
     assert np.array_equal(origins[address],expected)
   cases.append(dict(rows=rows,cols=cols,all_raw_indices_bijective=True,all16_rank49_virtual_plane_aliases_exact=True,bytes=rows*cols))
 # Independently evaluate rank7 Winograd on contiguous four-output lane groups.
 cols=256;q=rng.integers(-127,128,(2,cols),dtype=np.int64);w=rng.integers(-128,128,(8,cols),dtype=np.int64)
 a0,a1,a2,a3=q[0,:128],q[0,128:],q[1,:128],q[1,128:]
 b0,b1,b2,b3=w[:4,:128].T,w[4:,:128].T,w[:4,128:].T,w[4:,128:].T
 s1=a2+a3;s2=s1-a0;s3=a0-a2;s4=a1-s2;t1=b1-b0;t2=b3-t1;t3=b3-b1;t4=t2-b2
 dot=lambda a,b:(a[:,None]*b).sum(axis=0)
 p0,p1,p2,p3,p4,p5,p6=[dot(a,b)for a,b in [(a0,b0),(a1,b2),(s4,b3),(a3,t4),(s1,t1),(s2,t2),(s3,t3)]]
 u2=p0+p5;u3=u2+p6;got=np.stack([np.concatenate([p0+p1,u2+p4+p2]),np.concatenate([u3-p3,u3+p4])]);assert np.array_equal(got,q@w.T)
 report=dict(complete=True,cases=cases,rank7_contiguous_lane_scalar_oracle_equal=True,raw_capacity_multiplier=1,rank49_virtual_pointer_formula='((ki/2)*2+ni%2)*plane+(group16*2+ni/2)*cols*2+(ki%2)*256; block offset512, group stride4*cols',source_hashes={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},wasm_execution_verified=False,performance_verified=False,scope='Host bijection/address and independent rank7 integer-root proof only. Shared-layout Wasm generators, production OriginalView and full paid fidelity remain unimplemented/unverified.')
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in [Path(__file__),d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,shape_cases=len(cases),rank49_aliases=16,raw_capacity_multiplier=1)))
if __name__=='__main__':main()
