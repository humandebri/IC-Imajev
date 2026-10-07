#!/usr/bin/env python3
"""Measure exact BF16 storage capacity without changing model or weights."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
def main():
 mpath=ROOT/'checkpoints/full-int8.manifest.json';pack=ROOT/'checkpoints/full-int8.pack';model=ROOT/'MODEL_LOCK.json';m=json.loads(mpath.read_text());assert sha(model)==m['model'];assert sha(pack)==m['pack_hash'];out=[]
 with pack.open('rb')as f:
  for t in m['tensors']:
   if t['dtype']!='f32':continue
   f.seek(t['offset']);raw=f.read(t['bytes']);assert len(raw)==t['bytes'];a=np.frombuffer(raw,dtype='<u4');out.append(dict(name=t['name'],values=int(a.size),exact_bf16=int(np.count_nonzero((a&65535)==0)),tensor_sha256=hashlib.sha256(raw).hexdigest()))
 r=dict(model=m['model'],pack_hash=m['pack_hash'],pack_verified=True,tensors=out,values=sum(t['values']for t in out),exact_bf16=sum(t['exact_bf16']for t in out),whole_tensor_savings_bytes=sum(t['values']*2 for t in out if t['values']==t['exact_bf16']),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),mpath,model]})
 d=ROOT/'artifacts/lora-exact-bf16-capacity-v1';d.mkdir(exist_ok=True);(d/'verified.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['pack_verified','values','exact_bf16','whole_tensor_savings_bytes']}))
if __name__=='__main__':main()
