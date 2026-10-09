#!/usr/bin/env python3
"""Measure partial MLP preparation and bounded fused completion + next Delta.

All inference runs as ordinary query. The host losslessly frames canister-produced
carry only. Same-module separate completions provide an independent byte oracle.
This diagnostic does not establish full-model query-count reduction.
"""
import argparse,hashlib,json,pathlib,re,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,encode
from prefix_inference import verify_module
from mlp_delta_carry import encode_request,NAME,HUFFMAN_NAME
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--huffman',action='store_true');ap.add_argument('--raw-threshold',type=float,default=0.);ap.add_argument('--partial-rows',type=int,default=256);ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--layers',default='0,1,3');a=ap.parse_args();NAME=HUFFMAN_NAME if a.huffman else NAME
assert 0<a.partial_rows<2560 and a.partial_rows%32==0
rows=a.partial_rows;extra=[]if rows==256 else[rows]
d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);source=ROOT/'artifacts/mlp-pipeline-v1-617';raw_report=(source/'report.json').read_bytes();report=json.loads(raw_report);prefix=ROOT/'artifacts/prefix_codec/full-state-layout-proof/prefix';prefix_report=json.loads((prefix/'report.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
for field in ['model','pack_hash']:assert report[field]==prefix_report[field]
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'client').glob('*.py'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','crates/imajev-runtime/tests/mlp_delta_carry_identity.rs']]+[pathlib.Path(__file__)]
hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths};wasm=sha((ROOT/a.wasm).read_bytes());t=Transport(report['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,report['pack_hash']);cases=[];query_count=0
try:
 verify_module(t,wasm)
 for layer in map(int,a.layers.split(',')):
  assert 0<=layer<31 and (layer+2)%4!=0
  q=next(q for q in report['queries']if q['op']=='mlp_prepare_down'and int(re.search(r'\.layers\.(\d+)\.',q['tensor']).group(1))==layer)
  path=source/'queries'/f'{q["index"]:06d}.request.bin';h,x=decode(path.read_bytes());n=h['dims'][0];c=n*2560
  _,original_state=decode((source/'queries'/f'{q["index"]:06d}.response.bin').read_bytes())
  calls={}
  def call(label,header,values=None,frame=None,profile=False):
   global query_count
   request=d/f'{layer:02d}-{label}.request.bin';out=d/f'{layer:02d}-{label}.response.bin';request.write_bytes(frame if frame is not None else encode(header,values));query_count+=1
   try:result=t.command(dict(diagnostics=bool(profile),op='profile'if profile else'step',input=str(request),output=str(out)))
   except RuntimeError as error:result={'error':str(error)}
   info=dict(result=result,request_frame_bytes=request.stat().st_size,request_sha256=sha(request.read_bytes()))
   if 'ok'in result:
    rh,rv=decode(out.read_bytes());info['response_sha256']=sha(out.read_bytes());calls[label]=info;return rh,rv
   calls[label]=info;return None,None
  full=dict(h,op='mlp_full_integer',encoding='bf16-block256-exact-v1')
  _,baseline=call('full-mlp',full,x);assert baseline is not None
  partial=dict(h,op='mlp_prepare_partial_down',dims=[n,2560]+extra)
  _,state=call('prepare-partial',partial,x);assert state is not None
  assert state.size==original_state.size
  assert state[c:].tobytes()==original_state[c:].tobytes()
  for token in range(n):
   assert state[token*2560:token*2560+rows].tobytes()==baseline[token*2560:token*2560+rows].tobytes()
   assert state[token*2560+rows:(token+1)*2560].tobytes()==original_state[token*2560+rows:(token+1)*2560].tobytes()
  finish=dict(h,op='mlp_finish_partial_integer',encoding=NAME,dims=[n,0]+extra)
  _,done=call('finish-partial',finish,frame=encode_request(finish,state,raw_threshold=a.raw_threshold));assert done is not None and done.tobytes()==baseline.tobytes()
  sp=prefix/'queries/states'/f'layer-{layer+1:02d}.npz'
  with np.load(sp,allow_pickle=False)as saved:conv=saved['conv'].copy();log=saved['delta_log'].copy()
  dr=dict(h,op='delta_full_log_integer',encoding='bf16-block256-exact-v1',tensor=f'model.language_model.layers.{layer+1}.linear_attn.in_proj_qkv.weight',dims=[n,32,45,0],scalars=[],aux=[])
  _,delta=call('next-delta',dr,np.concatenate([done[c:],conv.ravel(),log.ravel()]));assert delta is not None
  fused=dict(h,op='mlp_finish_delta_log_integer',encoding=NAME,dims=[n,45]+extra)
  frame=encode_request(fused,state,conv,log,raw_threshold=a.raw_threshold)
  _,result=call('fused',fused,frame=frame)
  passed=result is not None and result[:c].tobytes()==done[:c].tobytes()and result[c:].tobytes()==delta.tobytes()
  if result is not None:assert passed
  if passed:
   _,profiled=call('fused-profile',fused,frame=frame,profile=True);assert profiled is not None and profiled.tobytes()==result.tobytes()
  else:
   # Diagnostic scaling only: copy already canister-produced rows. The main
   # benchmark remains all 87 tokens and the failed query is recorded above.
   short=45;sc=short*2560;snq=short*9216
   sx=state[c+n*9216:c+n*9216+n*36].reshape(n,36)[:short].ravel()
   ax=state[c+n*9216+n*36:].reshape(n,64)[:short].ravel()
   cut=np.concatenate([state[:sc],state[c:c+snq],sx,ax])
   short_dr=dict(dr,dims=[short,32,45,0]);_,short_delta=call('short-delta',short_dr,np.concatenate([baseline[c:c+sc],conv.ravel(),log.ravel()]));assert short_delta is not None
   short_fused=dict(fused,dims=[short,45]+extra);_,short_value=call('short-fused-profile',short_fused,frame=encode_request(short_fused,cut,conv,log,raw_threshold=a.raw_threshold),profile=True)
   assert short_value is not None and short_value[:sc].tobytes()==baseline[:sc].tobytes() and short_value[sc:].tobytes()==short_delta.tobytes()
  row=dict(layer=layer,tokens=n,prefix=45,partial_rows=rows,raw_threshold=a.raw_threshold,source_request_sha256=sha(path.read_bytes()),source_report_sha256=sha(raw_report),prefix_state_sha256=sha(sp.read_bytes()),full_output_bitwise_equal=passed,calls=calls)
  cases.append(row);print(json.dumps(row),flush=True)
 verify_module(t,wasm);assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
 result=dict(scope=__doc__,canister=a.canister,wasm_sha256=wasm,source_hashes=hashes,ordinary_queries=query_count,cases=cases,all_fused_passed=all(c['full_output_bitwise_equal']for c in cases),goal_50_verified=False)
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
