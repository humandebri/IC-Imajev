#!/usr/bin/env python3
import hashlib,json,pathlib,struct
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=ROOT/'checkpoints/adapter/decision_readout.safetensors'
b=p.read_bytes();n,=struct.unpack('<Q',b[:8]);h=json.loads(b[8:8+n]);t=h['weight'];a,z=t['data_offsets'];w=np.frombuffer(b[8+n+a:8+n+z],dtype='<f4').reshape(t['shape'])
model=hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest()
# One immutable pack holds separate weight precision variants. Activation quantization is evaluated independently.
pack=bytearray();tensors=[]
for name,dtype,body in [('readout-f32','f32',w.astype('<f4').tobytes()),('readout-int8','int8',None)]:
 if body is None:
  scales=np.maximum(np.max(abs(w),axis=1)/127,np.finfo(np.float32).tiny).astype('<f4');q=np.clip(np.rint(w/scales[:,None]),-127,127).astype(np.int8);body=q.tobytes()+scales.tobytes()
 tensors.append({'name':name,'offset':len(pack),'rows':256,'cols':2560,'dtype':dtype,'bytes':len(body)});pack.extend(body)
path=ROOT/'checkpoints/readout.pack';path.write_bytes(pack)
manifest={'version':1,'model':model,'pack_hash':hashlib.sha256(pack).hexdigest(),'bytes':len(pack),'tensors':tensors}
(ROOT/'checkpoints/readout.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
