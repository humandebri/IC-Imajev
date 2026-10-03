#!/usr/bin/env python3
"""Projected Delta capture/reuse against real saved separated kernels."""
import argparse,hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode,Transport,atomic
from prefix_inference import verify_module

def old_reference(a,d,name,h,x,n,first,keep,conv,state):
 root=h['tensor'].removesuffix('.in_proj_qkv.weight');base=dict(h,version=1,encoding='bf16-exact',aux=[],scalars=[])
 def call(tag,r,values):
  request=d/f'{name}-oracle-{tag}.request.bin';reply=d/f'{name}-oracle-{tag}.response.bin';atomic(request,encode(r,values))
  subprocess.run([str(ROOT/a.old_native),str(request),str(reply),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True)
  return decode(reply.read_bytes())[1]
 def project(part,rows,start,total):
  tensor=root+'.'+part;return call(part+str(start),dict(base,op='lora_integer',tensor=tensor+'.weight',dims=[n,rows,2560,start],aux=[tensor+'.lora_A.weight',tensor+'.lora_B.weight'],scalars=[2.]),x).reshape(n,rows)
 # Separate original kernels; independent binary, original unprepared A products.
 mixed=np.concatenate([project('in_proj_qkv',4096,0,8192),project('in_proj_qkv',4096,4096,8192)],axis=1)
 z=project('in_proj_z',4096,0,4096).reshape(n,32,128)
 gates=call('gates',dict(base,op='delta_gates_integer',tensor=root+'.in_proj_a.weight',dims=[n]),x)
 parts=[];states=[]
 for head in [first,first+8]:
  idx=np.r_[head//2*128:(head//2+4)*128,2048+head//2*128:2048+(head//2+4)*128,4096+head*128:4096+(head+8)*128]
  window=np.concatenate([conv[:,idx],mixed[:,idx]])
  values=np.concatenate([window.ravel(),z[:,head:head+8].ravel(),gates[:n*32].reshape(n,32)[:,head:head+8].ravel(),gates[n*32:].reshape(n,32)[:,head:head+8].ravel(),state[head:head+8].ravel()])
  out=call('stage'+str(head),dict(base,op='delta_stage_bf16',tensor=root+'.conv1d.weight',dims=[n,8,head,int(keep)],aux=[root+'.norm.weight']),values)
  parts.append(out[:n*8*128].reshape(n,8,128))
  if keep:states.append(out[n*8*128:])
 indices=np.r_[first//2*128:(first//2+8)*128,2048+first//2*128:2048+(first//2+8)*128,4096+first*128:4096+(first+16)*128]
 window=np.concatenate([conv[:,indices],mixed[:,indices]])
 return np.concatenate([np.concatenate(parts,axis=1).ravel(),window[-3:].ravel(),*(states if keep else [])])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',default='attention-fusion-v1');ap.add_argument('--native',required=True);ap.add_argument('--canister');ap.add_argument('--wasm');ap.add_argument('--directory',required=True);ap.add_argument('--old-native',default='artifacts/strassen-wide/default-primitive');a=ap.parse_args()
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());rows=[];negative=[];sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest() if a.wasm else None
 t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']) if a.canister else None
 try:
  if t:verify_module(t,sha)
  for label in ['prefix','617','insufficient','maximum','normal']:
   directory=ROOT/f'artifacts/{a.baseline}-{label}';raw=(directory/'report.json').read_bytes();report=json.loads(raw)
   for layer in [0,30]:
    root=f'model.language_model.layers.{layer}.linear_attn';q=next(q for q in report['queries'] if q['tensor']==root+'.in_proj_qkv.weight');h,x=decode((directory/'queries'/f'{q["index"]:06d}.request.bin').read_bytes());n=h['dims'][0];assert x.size==n*2560
    keep=label=='prefix'
    if label in ('prefix','normal'):conv=np.zeros((3,8192),np.float32);state=np.zeros((32,128,128),np.float32)
    else:
     with np.load(ROOT/f'artifacts/{a.baseline}-prefix/queries/states/layer-{layer:02d}.npz') as f:conv=f['conv'].copy();state=f['delta'].copy()
    oq=next(q for q in report['queries'] if q['tensor']==root+'.out_proj.weight');_,out=decode((directory/'queries'/f'{oq["index"]:06d}.request.bin').read_bytes());out=out.reshape(n,32,128)
    with np.load(directory/f'queries/states/layer-{layer:02d}.npz') as f:final_conv=f['conv'].copy();final_state=f['delta'].copy() if keep else None
    saved=None
    for first in [0,16]:
     name=f'{label}-layer{layer}-head{first}';indices=np.concatenate([np.arange(first//2*128,(first//2+8)*128),np.arange(2048+first//2*128,2048+(first//2+8)*128),np.arange(4096+first*128,4096+(first+16)*128)])
     values=np.concatenate([x if saved is None else saved,conv[:,indices].ravel(),state[first:first+16].ravel()])
     header=dict(h,op='delta_project_capture' if saved is None else 'delta_project_reuse',dims=[n,16,first,int(keep)],aux=[],scalars=[],encoding='bf16-block256-exact-v1')
     request=d/f'{name}.request.bin';native=d/f'{name}.native.bin';reply=d/f'{name}.response.bin';atomic(request,encode(header,values))
     if t:
      measured=t.command(dict(op='step',input=str(request),output=str(reply)));body=reply.read_bytes()
     else:
      subprocess.run([str(ROOT/a.native),str(request),str(native),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True);body=native.read_bytes()
     _,got=decode(body);expected=np.concatenate([out[:,first:first+16].ravel(),final_conv[:,indices].ravel(),*([final_state[first:first+16].ravel()] if keep else [])])
     platform_difference=int(np.count_nonzero(got[:len(expected)].view(np.uint32)!=expected.view(np.uint32)))
     if t:
      np.testing.assert_array_equal(got[:len(expected)].view(np.uint32),expected.view(np.uint32),err_msg=name)
     else:
      independent=old_reference(a,d,name,h,x,n,first,keep,conv,state)
      np.testing.assert_array_equal(got[:len(independent)].view(np.uint32),independent.view(np.uint32),err_msg=name+' independent native')
     if saved is None:
      saved=got[len(expected):];assert saved.size==n*2762
     else:assert len(got)==len(expected)
     row=dict(name=name,op=header['op'],dims=header['dims'],request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),reply_sha256=hashlib.sha256(body).hexdigest(),reference_bitwise_equal=True,differences_to_saved_wasm_values=platform_difference,source_report_sha256=hashlib.sha256(raw).hexdigest())
     if t:row['measurement']=measured;row['wasm_old_wasm_bitwise_equal']=True
     else:row['native_old_native_bitwise_equal']=True
     rows.append(row);print(json.dumps(row),flush=True)
  if t:
   h,x=decode((d/'617-layer0-head16.request.bin').read_bytes());integer=87*2560
   variants={}
   for name,index,value in [('fractional',0,.5),('reserved',0,-128.),('fractional-last',integer-1,.5),('reserved-last',integer-1,-128.),('zero-scale',integer,0.),('negative-scale',integer,-1.)]:
    values=x.copy();values[index]=value;variants[name]=(h.copy(),values)
   invalid=h.copy();invalid['dims']=[87,16,1,0];variants['odd-first']=(invalid,x)
   for name,(r,values) in variants.items():
    request=d/f'bad-{name}.bin';atomic(request,encode(r,values))
    try:t.command(dict(op='step',input=str(request),output=str(d/f'bad-{name}.response.bin')))
    except RuntimeError as e:negative.append(dict(name=name,error=str(e)))
    else:raise AssertionError('Accepted '+name)
   verify_module(t,sha)
 finally:
  if t:t.close()
 result=dict(baseline=a.baseline,wasm_sha256=sha,native_sha256=hashlib.sha256((ROOT/a.native).read_bytes()).hexdigest(),old_native_sha256=hashlib.sha256((ROOT/a.old_native).read_bytes()).hexdigest(),cases=rows,negative=negative,successful_ordinary_queries=len(rows) if t else 0,rejected_ordinary_queries=len(negative),certified_module_reads=2 if t else 0,scope='Two Delta layers per five real sequences; full graph and general accuracy not implied',script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
