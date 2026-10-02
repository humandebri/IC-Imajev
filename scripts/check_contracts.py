#!/usr/bin/env python3
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from transport import Transport,encode,atomic
from checkpoint import Checkpoint
m=json.loads((ROOT/'checkpoints/readout.manifest.json').read_text());directory=ROOT/'artifacts/contracts';t=Transport(m['model'],'http://localhost:8001/','4caro-hl777-77775-aaaba-cai',str(ROOT/'artifacts/imajev-local.pem'),directory,m['pack_hash']);r=json.loads((ROOT/'artifacts/reference-first.json').read_text())['records'][0]
h={'version':1,'model':m['model'],'pack_hash':m['pack_hash'],'input_hash':r['input_sha256'],'step':0,'op':'matmul','tensor':'readout-f32','dims':[1,256,2560],'scalars':[]};path=directory/'typed.bin';atomic(path,encode(h,r['hidden']))
d=t.command({'op':'decision','input':str(path),'options':r['options']});assert d['ok']['decision']['value']==r['result']['value'];report={'typed_choice':d,'negative_tests':[],'wasm_sha256':hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()}
for name,header,data in [('wrong_pack',{**h,'pack_hash':'0'*64},encode({**h,'pack_hash':'0'*64},r['hidden'])),('bad_checksum',h,encode(h,r['hidden'])[:-1]+b'\x00')]:
 p=directory/f'{name}.bin';atomic(p,data)
 try:t.command({'op':'step','input':str(p),'output':str(directory/'rejected.bin')})
 except RuntimeError as e:report['negative_tests'].append({'case':name,'rejected':True,'message':str(e)})
 else:raise AssertionError(name)
try:t.command({'op':'decision','input':str(path),'options':['yes','yes']})
except RuntimeError as e:report['negative_tests'].append({'case':'duplicate_options','rejected':True,'message':str(e)})
else:raise AssertionError('duplicate')
w=Checkpoint(ROOT/'checkpoints/adapter/decision_readout.safetensors').tensor('weight');x=np.array(r['hidden'],dtype=np.float32);y=t.run('matmul',x,[1,4,2560,8],tensor='readout-f32',input_hash=r['input_sha256']);error=float(abs(y-w[8:12]@x).max());assert error<1e-4;report['nonzero_row_tile']={'max_error':error,'stable_read_bytes':t.measurements[-1]['ok']['stable_read_bytes']}
request=directory/'000000.request.bin';response=directory/'replay.bin';a=t.command({'op':'step','input':str(request),'output':str(response)});first=response.read_bytes();b=t.command({'op':'step','input':str(request),'output':str(response)});assert response.read_bytes()==first;report['query_replay_identical']=True
report['queries']=t.measurements;t.close();(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
