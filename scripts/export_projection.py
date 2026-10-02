#!/usr/bin/env python3
"""Real layer-0 QKV base + separate F32 LoRA; never silently merge/quantize adapter."""
import hashlib,json,pathlib,sys
import numpy as np
from checkpoint import Checkpoint
ROOT=pathlib.Path(__file__).resolve().parents[1];base=Checkpoint(ROOT/'checkpoints/base/model.safetensors-00002-of-00002.safetensors');adapter=Checkpoint(ROOT/'checkpoints/adapter/adapter_model.safetensors')
prefix='model.language_model.layers.0.linear_attn.in_proj_qkv';a='base_model.model.'+prefix
w=base.tensor(prefix+'.weight')[:256];A=adapter.tensor(a+'.lora_A.weight');B=adapter.tensor(a+'.lora_B.weight')[:256]
pack=bytearray();ts=[]
for name,arr in [('qkv-base-256',w),('qkv-lora-A',A),('qkv-lora-B-256',B)]:
 b=arr.astype('<f4').tobytes();ts.append({'name':name,'offset':len(pack),'rows':arr.shape[0],'cols':arr.shape[1],'dtype':'f32','bytes':len(b)});pack.extend(b)
scale=np.maximum(np.max(abs(w),axis=1)/127,np.finfo(np.float32).tiny).astype('<f4');b=np.clip(np.rint(w/scale[:,None]),-127,127).astype(np.int8).tobytes()+scale.tobytes();ts.append({'name':'qkv-base-int8-256','offset':len(pack),'rows':256,'cols':2560,'dtype':'int8','bytes':len(b)});pack.extend(b)
model=hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest();p=ROOT/'checkpoints/projection.pack';p.write_bytes(pack);(ROOT/'checkpoints/projection.manifest.json').write_text(json.dumps({'version':1,'model':model,'pack_hash':hashlib.sha256(pack).hexdigest(),'bytes':len(pack),'tensors':ts},indent=2)+'\n');print('pack bytes',len(pack))
