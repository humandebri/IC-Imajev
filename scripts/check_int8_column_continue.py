#!/usr/bin/env python3
"""Real Delta gated inputs: exact INT8/F32 column carry and final original LoRA rounding."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();wasm=sha(ROOT/a.wasm);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());cases=[];profiles=[];rejected=[]
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__),ROOT/'Cargo.lock',ROOT/'MODEL_LOCK.json',ROOT/'checkpoints/full-int8.manifest.json'];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths}
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='bf16-block256-exact-v1',frame_version=3)
try:
 verify_module(t,wasm)
 for label in ['prefix','617','insufficient','maximum']:
  source=ROOT/f'artifacts/column16-v1-{label}';report=json.loads((source/'report.json').read_bytes());assert report['model']==m['model']and report['pack_hash']==m['pack_hash']
  for layer in [0,30]:
   tensor=f'model.language_model.layers.{layer}.linear_attn.out_proj.weight';q=next(q for q in report['queries']if q['op']=='lora_integer'and q['tensor']==tensor);request=source/'queries'/f"{q['index']:06d}.request.bin";response=source/'queries'/f"{q['index']:06d}.response.bin";h,x=decode(request.read_bytes());_,expected=decode(response.read_bytes());n=h['dims'][0];assert h['dims']==[n,2560,4096,0];values=x.reshape(n,4096);name=tensor[:-len('.weight')];adapter=[name+'.lora_A.weight',name+'.lora_B.weight']
   full_base=t.run('int8_matmul',x,[n,2560,4096,0],tensor=tensor,input_hash=h['input_hash']);full_a=t.run('matmul',x,[n,64,4096],tensor=adapter[0],input_hash=h['input_hash']);full_lora=t.run('lora_integer',x,[n,2560,4096,0],[2.],tensor=tensor,input_hash=h['input_hash'],aux=adapter);assert full_lora.tobytes()==expected.tobytes()
   if label=='617'and layer==0:sample=(h,x,tensor,adapter,full_base,full_a)
   for chunks in [[3072,1024],[2816,1280],[2048,2048],[1024,1024,1024,1024]]:
    base=np.zeros(n*2560,np.float32);ax=np.zeros(n*64,np.float32);begin=0;start=len(t.measurements);frames=[]
    for count in chunks:
     incoming=values[:,begin:begin+count].copy().ravel();base=t.run('linear_integer_k_continue',np.concatenate([incoming,base]),[n,2560,4096,begin,count],tensor=tensor,input_hash=h['input_hash']);frames.append(t.index-1)
     ax=t.run('matmul_k_continue',np.concatenate([incoming,ax]),[n,64,4096,begin,count],tensor=adapter[0],input_hash=h['input_hash']);begin+=count
    assert begin==4096 and base.tobytes()==full_base.tobytes()and ax.tobytes()==full_a.tobytes()
    y=t.run('linear_integer_k_finish',np.concatenate([base,ax]),[n,2560,4096],[2.],tensor=tensor,input_hash=h['input_hash'],aux=adapter);frames.append(t.index-1);assert y.tobytes()==expected.tobytes();calls=t.measurements[start:]
    row=dict(label=label,layer=layer,tokens=n,chunks=chunks,base_bitwise_equal=True,a_bitwise_equal=True,lora_bitwise_equal=True,source_request_sha256=sha(request),source_response_sha256=sha(response),ordinary_queries=len(calls),instructions=sum(c['ok']['instructions']for c in calls),candid_bytes=sum(c['ok']['request_bytes']+c['ok']['reply_bytes']for c in calls),maximum_query_instructions=max(c['ok']['instructions']for c in calls),calls=calls);cases.append(row);print(json.dumps(row),flush=True)
    if label=='617'and layer==0 and chunks==[2816,1280]:
     for index in frames:
      req=d/f'{index:06d}.request.bin';out=d/f'{index:06d}.profile.response.bin';metric=t.command(dict(op='profile',input=str(req),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==decode((d/f'{index:06d}.response.bin').read_bytes())[1].tobytes();profiles.append(metric)
 h,x,tensor,adapter,full_base,full_a=sample;n=h['dims'][0];part=x.reshape(n,4096)[:,:256].copy().ravel();initial=np.zeros(n*2560,np.float32);bad_initial=initial.copy();bad_initial[0]=-0.;bad_part=part.copy();bad_part[0]=np.float32(.1234567);finish=np.concatenate([full_base,full_a])
 bad=[('initial-minus-zero','linear_integer_k_continue',np.concatenate([part,bad_initial]),[n,2560,4096,0,256],[],[]),('non-bf16-input','linear_integer_k_continue',np.concatenate([bad_part,initial]),[n,2560,4096,0,256],[],[]),('unaligned-column','linear_integer_k_continue',np.concatenate([part,initial]),[n,2560,4096,1,256],[],[]),('finish-wrong-rank-size','linear_integer_k_finish',finish[:-1],[n,2560,4096],[2.],adapter),('finish-wrong-alpha','linear_integer_k_finish',finish,[n,2560,4096],[1.],adapter),('finish-wrong-adapter','linear_integer_k_finish',finish,[n,2560,4096],[2.],[adapter[0],adapter[0]])]
 for name,op,values,dims,scalars,aux in bad:
  try:t.run(op,values,dims,scalars,tensor=tensor,aux=aux,input_hash=h['input_hash'])
  except RuntimeError as e:rejected.append(dict(name=name,error=str(e)))
  else:raise AssertionError('Accepted '+name)
 verify_module(t,wasm);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths};assert len(cases)==32 and len(profiles)==3
 result=dict(scope=__doc__,wasm_sha256=wasm,model=m['model'],pack_hash=m['pack_hash'],source_hashes=hashes,cases=cases,profiles=profiles,regular_diagnostic_queries=len(t.measurements),profile_diagnostic_queries=len(profiles),rejected_ordinary_queries=len(rejected),rejected=rejected,goal_50_verified=False);(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
