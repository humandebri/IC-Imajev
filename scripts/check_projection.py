#!/usr/bin/env python3
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport
ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://localhost:8001/');ap.add_argument('--canister',default='4fbx2-kt777-77775-aaabq-cai');ap.add_argument('--upload',action='store_true');args=ap.parse_args()
m=json.loads((ROOT/'checkpoints/projection.manifest.json').read_text());t=Transport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/projection-check',m['pack_hash']);report={'scope':'First 256 output rows of real layer-0 QKV projection; unmerged LoRA rank64 scale2; not an entire block or model','model':m['model'],'pack_hash':m['pack_hash'],'wasm_sha256':hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),'url':args.url,'canister':args.canister,'query_cache_controlled':False}
if args.upload:report['upload']=t.upload(ROOT/'checkpoints/projection.manifest.json',ROOT/'checkpoints/projection.pack')
d=np.load(ROOT/'artifacts/projection-layer0.npz');rows=[]
for n in [32,64,128,132]:
 for weight,activation in [('qkv-base-256','f32'),('qkv-base-int8-256','f32'),('qkv-base-256','int8')]:
  x=d['x'][:n].copy()
  if activation=='int8':
   scale=np.maximum(np.max(abs(x),axis=1)/127,np.finfo(np.float32).tiny);x=np.clip(np.rint(x/scale[:,None]),-127,127).astype(np.int8).astype(np.float32)*scale[:,None]
  base=t.run('matmul',x,[n,256,2560],tensor=weight)
  a=t.run('matmul',x,[n,64,2560],tensor='qkv-lora-A');z=t.run('matmul',a,[n,256,64],tensor='qkv-lora-B-256')
  y=t.run('lora_finish',np.concatenate([base,z]),[n*256],[2.0]).reshape(n,256)
  error=abs(y-d['output'][:n]);rows.append({'tokens':n,'weight':weight,'activation':activation,'max_error':float(error.max()),'mean_error':float(error.mean()),'rmse':float(np.sqrt(np.mean(error**2)))})
  print(rows[-1],flush=True)
report['comparisons']=rows;report['queries']=t.measurements;report['total_instructions']=sum(q['ok']['instructions']for q in t.measurements);report['total_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes']for q in t.measurements);report['total_wall_seconds']=sum(q['wall_seconds']for q in t.measurements);t.close();(ROOT/'artifacts/projection-check/report.json').write_text(json.dumps(report,indent=2)+'\n')
