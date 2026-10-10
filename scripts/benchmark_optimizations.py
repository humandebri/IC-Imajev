#!/usr/bin/env python3
"""A/B real operands on ordinary queries; no weight or activation quantization."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,atomic
out=ROOT/'artifacts/optimization-check';out.mkdir(parents=True,exist_ok=True)
def transport(pack,canister):
 m=json.loads((ROOT/f'checkpoints/{pack}.manifest.json').read_text())
 return Transport(m['model'],'http://localhost:8001/',canister,str(ROOT/'artifacts/imajev-local.pem'),out/pack,m['pack_hash'])
def summary(queries):
 return dict(calls=len(queries),instructions=sum(q['ok']['instructions'] for q in queries),candid_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in queries),wall_seconds=sum(q['wall_seconds'] for q in queries),stable_read_bytes=sum(q['ok'].get('stable_read_bytes',0) for q in queries),max_query_instructions=max(q['ok']['instructions'] for q in queries))
p=transport('projection','4fbx2-kt777-77775-aaabq-cai');d=np.load(ROOT/'artifacts/projection-layer0.npz');rows=[]
for n in [4,5,32,64,128,132]:
 x=d['x'][:n];outputs={};metrics={};intermediates={}
 for mode in ['scalar','simd','fused']:
  start=len(p.measurements)
  if mode=='fused':
   y=p.run('lora_project',x,[n,256,2560],[2.0],tensor='qkv-base-256',aux=['qkv-lora-A','qkv-lora-B-256'])
  else:
   op='matmul_reference' if mode=='scalar' else 'matmul'
   base=p.run(op,x,[n,256,2560],tensor='qkv-base-256')
   a=p.run(op,x,[n,64,2560],tensor='qkv-lora-A')
   z=p.run(op,a,[n,256,64],tensor='qkv-lora-B-256')
   intermediates[mode]=(base,a,z)
   if mode=='simd':
    for before,after in zip(intermediates['scalar'],intermediates['simd']):assert np.array_equal(before.view(np.uint32),after.view(np.uint32)),(n,'intermediate changed bits')
   y=p.run('lora_finish',np.concatenate([base,z]),[n*256],[2.])
  outputs[mode]=y;metrics[mode]=summary(p.measurements[start:])
  assert np.array_equal(y.view(np.uint32),outputs['scalar'].view(np.uint32)),(n,mode,'changed bits')
 # Output-row tiling must slice both base and LoRA B, and return token-major output.
 if n==132:
  tiles=[];start=len(p.measurements)
  for row,width in [(0,127),(127,129)]:
   tiles.append(p.run('lora_project',x,[n,width,2560,row],[2.],tensor='qkv-base-256',aux=['qkv-lora-A','qkv-lora-B-256']).reshape(n,width))
  tiled=np.concatenate(tiles,axis=1).ravel();assert np.array_equal(tiled.view(np.uint32),outputs['scalar'].view(np.uint32))
  metrics['row_split']=summary(p.measurements[start:])
 rows.append(dict(tokens=n,bitwise_equal=True,all_three_f32_products_bitwise_equal=True,official_bf16_max_error=float(abs(outputs['fused'].reshape(n,256)-d['output'][:n]).max()),variants=metrics));print(json.dumps(rows[-1]),flush=True)
p.close()
report=dict(scope='Real layer-0 256-row QKV base+LoRA; not full-model inference',url='http://localhost:8001/',query_cache_controlled=False,communication_scope='Candid request+reply only',instruction_scope='Inside handler; excludes CDK Candid decode/encode.',wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),projection=rows,projection_queries=p.measurements)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('saved',out/'report.json')
