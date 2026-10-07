#!/usr/bin/env python3
"""Read-only inspection of all actual LoRA F32 weights for exact sparse/duplicate structure."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/lora-structure-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False);manifest=ROOT/'checkpoints/full-int8.manifest.json';m=json.loads(manifest.read_text());pack=ROOT/'checkpoints/full-int8.pack';results=[]
 for t in m['tensors']:
  if 'lora_'not in t['name']:continue
  assert t['dtype']=='f32' and t['bytes']==t['rows']*t['cols']*4
  values=np.memmap(pack,mode='r',dtype='<f4',offset=t['offset'],shape=(t['rows'],t['cols']));bits=values.view('<u4');zero=(bits&0x7fffffff)==0;assert np.all(np.isfinite(values));rows=t['rows'];cols=t['cols'];zh=(zero[:rows//4*4].reshape(rows//4,4,cols).all(axis=1))
  row_hash=[hashlib.sha256(v.tobytes()).digest()for v in bits];column_hash=[hashlib.sha256(bits[:,i].tobytes()).digest()for i in range(cols)];results.append(dict(name=t['name'],rows=rows,cols=cols,weight_sha256=hashlib.sha256(values.tobytes()).hexdigest(),zero_elements=int(zero.sum()),elements=int(values.size),zero_rows=int(zero.all(axis=1).sum()),zero_columns=int(zero.all(axis=0).sum()),zero_output4_vectors=int(zh.sum()),output4_vectors=int(zh.size),duplicate_rows=rows-len(set(row_hash)),duplicate_columns=cols-len(set(column_hash))))
  del values,bits,zero,zh
 total={k:sum(t[k]for t in results)for k in ['zero_elements','elements','zero_rows','zero_columns','zero_output4_vectors','output4_vectors','duplicate_rows','duplicate_columns']};r=dict(tensors=len(results),total=total,cases=results,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),manifest,ROOT/'MODEL_LOCK.json']},pack_hash=m['pack_hash'],scope='All400 actual LoRA tensors; numeric zero includes signed zero, duplicate rows/columns require exact stored F32 bits. Read-only, no kernel performance claim.');assert len(results)==400;(D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(tensors=len(results),total=total)))
if __name__=='__main__':main()
