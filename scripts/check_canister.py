#!/usr/bin/env python3
"""Actual decision-head and recurrent parity on owner-only ordinary queries."""
import argparse,hashlib,json,pathlib,sys,subprocess,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode,atomic
ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://localhost:8001/');ap.add_argument('--canister',default='4caro-hl777-77775-aaaba-cai');ap.add_argument('--upload',action='store_true');args=ap.parse_args()
manifest=ROOT/'checkpoints/readout.manifest.json';pack=ROOT/'checkpoints/readout.pack';m=json.loads(manifest.read_text());directory=ROOT/'artifacts/canister-check';directory.mkdir(parents=True,exist_ok=True)
t=Transport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),directory,m['pack_hash'])
report={'scope':'Readout and one DeltaNet head from real official layer-0 operands; NOT full 4B canister inference.','model':m['model'],'pack_hash':m['pack_hash'],'url':args.url,'canister':args.canister,'wasm_sha256':hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),'query_cache_controlled':False,'communication_scope':'Candid request and reply bytes, excludes HTTP/CBOR/signature envelope','instruction_scope':'handler includes state validation, stable reads, tensor unpack, kernels and inner state serialization; excludes CDK Candid decoding/encoding'}
if args.upload:report['upload']=t.upload(manifest,pack)
records=json.loads((ROOT/'artifacts/reference-orders.json').read_text())['records'];results=[]
for c in records:
 x=np.array(c['hidden'],dtype=np.float32);indices=c['metadata']['readout_indices'];expected=np.array(list(c['raw_result']['raw_logits'].values()))
 for weight,state_mode in [('readout-f32','f32'),('readout-int8','f32'),('readout-f32','int8')]:
  inp=x.copy()
  if state_mode=='int8':
   scale=max(float(abs(x).max())/127,np.finfo(np.float32).tiny);inp=(np.clip(np.rint(x/scale),-127,127).astype(np.int8).astype(np.float32)*scale)
  logits=t.run('matmul',inp,[1,256,2560],tensor=weight,input_hash=c['input_sha256'])[indices]
  probabilities=t.run('softmax',logits,scalars=[1.3051569717552742],input_hash=c['input_sha256'])
  winner=int(np.argmax(logits));labels=c['options']+['__unknown__'];expected_label='__unknown__' if c['result']['value'] is None else c['result']['value']
  results.append({'id':c['id'],'offset':c['offset'],'weight':weight,'activation':state_mode,'logits':logits.tolist(),'probabilities':probabilities.tolist(),'max_logit_error':float(abs(logits-expected).max()),'max_probability_error':float(abs(probabilities-np.array(list(c['result']['scores'].values()))).max()),'label':labels[winner],'reference_label':expected_label,'gold':c['gold'],'matches_reference':labels[winner]==expected_label,'typed_output_valid':bool(np.isfinite(probabilities).all() and abs(probabilities.sum()-1)<1e-6 and min(probabilities)>=0 and max(probabilities)<=1)})
 print(c['id'],c['offset'],'readout complete',flush=True)
report['readout']=results
if (ROOT/'artifacts/delta-layer0.npz').exists():
 d=np.load(ROOT/'artifacts/delta-layer0.npz');head=0;n,dk=d['q'].shape[1],d['q'].shape[3];dv=d['v'].shape[3]
 q=d['q'][0,:,head//2];k=d['k'][0,:,head//2];v=d['v'][0,:,head];g=d['g'][0,:,head];beta=d['beta'][0,:,head];gold=d['output'][0,:,head]
 inputs=np.concatenate([q.ravel(),k.ravel(),v.ravel(),g,beta,np.zeros(dk*dv,dtype=np.float32)])
 y=t.run('delta',inputs,[n,dk,dv]);full=y[:n*dv].reshape(n,dv);state=y[n*dv:]
 # Client-held recurrent state continues correctly across independent ordinary queries.
 split=n//2;s=np.zeros(dk*dv,dtype=np.float32);parts=[]
 for a,z in [(0,split),(split,n)]:
  inp=np.concatenate([q[a:z].ravel(),k[a:z].ravel(),v[a:z].ravel(),g[a:z],beta[a:z],s]);out=t.run('delta',inp,[z-a,dk,dv]);parts.append(out[:(z-a)*dv]);s=out[(z-a)*dv:]
 continued=np.concatenate(parts).reshape(n,dv)
 report['delta']={'tokens':n,'key_dim':dk,'value_dim':dv,'max_f32_vs_official_bf16_error':float(abs(full-gold).max()),'split_output_max_error':float(abs(full-continued).max()),'split_state_max_error':float(abs(state-s).max())}
 # Native and Wasm execute the same recorded request, without trusting a Python duplicate.
 index=t.index-3;req=directory/f'{index:06d}.request.bin';native=directory/'native-delta.bin';subprocess.run([str(ROOT/'target/release/primitive'),str(req),str(native)],check=True)
 _,values=decode(native.read_bytes());report['delta']['native_vs_wasm_max_error']=float(abs(values-y).max())
report['queries']=t.measurements;report['total_instructions']=sum(q['ok']['instructions'] for q in t.measurements);report['total_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in t.measurements);report['total_wall_seconds']=sum(q['wall_seconds'] for q in t.measurements)
t.close();(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['readout','queries']},indent=2))
