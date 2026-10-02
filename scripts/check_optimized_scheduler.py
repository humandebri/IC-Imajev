#!/usr/bin/env python3
import json,pathlib,sys,hashlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
m=json.loads((ROOT/'checkpoints/projection.manifest.json').read_text());d=np.load(ROOT/'artifacts/projection-layer0.npz');t=Transport(m['model'],'http://localhost:8001/','4fbx2-kt777-77775-aaabq-cai',str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/optimized-scheduler',m['pack_hash'])
x=d['x'];reference=t.run('lora_project',x,[132,256,2560],[2.],tensor='qkv-base-256',aux=['qkv-lora-A','qkv-lora-B-256']).reshape(132,256)
y=t.project(x,256,2560,64,'qkv-base-256','qkv-lora-A','qkv-lora-B-256',row_cap=129,token_cap=64)
assert np.array_equal(reference.view(np.uint32),y.view(np.uint32))
report=dict(scope='Token+output-row split of independent QKV linear projection, not a full layer',bitwise_equal=True,wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),queries=t.measurements)
(ROOT/'artifacts/optimized-scheduler/report.json').write_text(json.dumps(report,indent=2)+'\n');t.close();print('scheduler passed',len(report['queries'])-1,'split queries')
