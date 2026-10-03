#!/usr/bin/env python3
"""Fixed-weight proof for grouping signed I8 products in exact I16 accumulators.
No new inference and no quantization; measurements are static eligibility only.
"""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());names=['model.language_model.layers.3.self_attn.q_proj.weight','model.language_model.layers.0.linear_attn.in_proj_qkv.weight','model.language_model.layers.0.mlp.gate_proj.weight','model.language_model.layers.0.mlp.down_proj.weight'];cases=[]
 with (ROOT/'checkpoints/full-int8.pack').open('rb') as f:
  for name in names:
   t=next(t for t in m['tensors'] if t['name']==name);rows,cols=t['rows'],t['cols'];assert t['dtype']=='int8' and cols%256==0 and rows%8==0
   f.seek(t['offset']);raw=f.read(rows*cols);assert len(raw)==rows*cols
   w=np.frombuffer(raw,np.int8).astype(np.int16).reshape(rows,cols//256,16,16);pair=np.abs(w[:,:,:,:8])+np.abs(w[:,:,:,8:]);results={}
   for group in [1,2,4,8,16]:
    sums=pair.reshape(rows,cols//256,16//group,group,8).sum(axis=3,dtype=np.int32)
    # Every lane, partial group and one of eight output rows must fit.
    bounds=sums.reshape(rows//8,8,cols//256,16//group,8).max(axis=(1,3,4));safe=bounds<=32767//127
    results[str(group)]=dict(safe_tiles=int(safe.sum()),tiles=int(safe.size),fraction=float(safe.mean()),maximum_lane_abs_weight_sum=int(bounds.max()))
   cases.append(dict(name=name,rows=rows,cols=cols,weight_bytes_sha256=hashlib.sha256(raw).hexdigest(),groups=results));print(json.dumps(cases[-1]))
 out=ROOT/'artifacts/prepared-rope/i16-weight-bounds.json';out.write_text(json.dumps(dict(scope='Static exact-I16 safety eligibility only; no Wasm speed/instruction/full-model claim',model=m['model'],pack_hash=m['pack_hash'],activation_bound=127,maximum_weight_lane_sum=258,cases=cases),indent=2)+'\n')
if __name__=='__main__':main()
