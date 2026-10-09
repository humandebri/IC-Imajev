#!/usr/bin/env python3
"""Measure an actual MLP->KV->Q/attention->MLP carry chain against frozen canister outputs."""
import argparse,hashlib,json,pathlib,sys,zipfile,shutil
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
from mlp_stream_codec import NAME as MLP_NAME
from attention_mlp_stream_codec import NAME,C,KV,encode_request
from transport import is_instruction_limit
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--q4',action='store_true');ap.add_argument('--cases',default='617,insufficient,maximum');ap.add_argument('--layers',default='2');ap.add_argument('--front',type=int,default=1024);ap.add_argument('--begins',default='4096,4352,4608,4864,5120');a=ap.parse_args()
d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);assert not(d/'report.json').exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());module=sha(ROOT/a.wasm)
paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths}
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec=MLP_NAME,frame_version=3);cases=[]
def query(root,report,op,layer):
 q=next(q for q in report['queries']if q['op']==op and f'.layers.{layer}.'in q['tensor']);request=root/'queries'/f"{q['index']:06d}.request.bin";response=root/'queries'/f"{q['index']:06d}.response.bin";h,x=decode(request.read_bytes());_,expected=decode(response.read_bytes());return h,x,expected,request,response
try:
 verify_module(t,module)
 for label in a.cases.split(','):
  root=ROOT/'artifacts/prefix_codec/full-roll-review-base-proof-v7'/label;report=json.loads((root/'report.json').read_text())
  for layer in map(int,a.layers.split(',')):
   h,x,both,req,res=query(root,report,'mlp_full_integer',layer);n=h['dims'][0]
   _,_,att,areq,ares=query(root,report,'attention_full_integer',layer+1);kv=att[n*C:]
   mh,mx,next_both,mreq,mres=query(root,report,'mlp_full_integer',layer+1)
   with np.load(ROOT/'artifacts/prefix_codec/full-roll-review-base-proof-v7/prefix/queries/states'/f'layer-{layer+1:02d}.npz',allow_pickle=False)as z:prefix=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()])
   t.wire_codec=MLP_NAME;carry=t.run('mlp_stream_prepare',x,[n,0,a.front],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash'])
   header=dict(h,version=3,step=t.index,encoding=NAME,op='mlp_complete_attention_kv_q4'if a.q4 else'mlp_complete_attention_kv',dims=[n,a.front,9216-a.front,45]);packet=encode_request(header,carry,prefix);index=t.index
   try:y=t._run_encoded(header,packet)
   except RuntimeError as error:
    if not is_instruction_limit(error):raise
    shutil.copyfile(d/f'{index:06d}.request.bin',d/f'failed-{label}-layer{layer}-first.request.bin')
    cases.append(dict(label=label,layer=layer,tokens=n,stage='mlp_complete_kv',success=False,error=str(error),request_sha256=sha(d/f'{index:06d}.request.bin')));continue
   assert y[:2*n*C].tobytes()==both.tobytes();assert y[2*n*C:n*(2*C+KV)].tobytes()==kv.tobytes();first=t.measurements[-1]
   out=d/f'{index:06d}.profile.response.bin';profile=t.command(dict(diagnostics=True,op='profile',input=str(d/f'{index:06d}.request.bin'),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==y.tobytes()
   for begin in map(int,a.begins.split(',')):
    bh=dict(mh,version=3,step=t.index,encoding=NAME,op='attention_finish_mlp_front_q4'if a.q4 else'attention_finish_mlp_front',dims=[n,begin,45]);bi=t.index;bp=encode_request(bh,y,prefix)
    try:result=t._run_encoded(bh,bp)
    except RuntimeError as error:
     if not is_instruction_limit(error):raise
     shutil.copyfile(d/f'{bi:06d}.request.bin',d/f'failed-{label}-layer{layer+1}-begin{begin}.request.bin')
     row=dict(label=label,layer=layer+1,tokens=n,begin=begin,stage='attention_mlp_front',success=False,error=str(error),request_sha256=sha(d/f'{bi:06d}.request.bin'));cases.append(row);print(json.dumps(row),flush=True);continue
    call=t.measurements[-1];assert result[-n*KV:].tobytes()==kv.tobytes();state=result[:-n*KV]
    expected=t.run('mlp_stream_prepare',mx,[n,0,begin],[2.,1e-6],tensor=mh['tensor'],aux=mh['aux'],input_hash=mh['input_hash']);assert state.tobytes()==expected.tobytes()
    final=t.run('mlp_stream_complete',state,[n,begin,9216-begin],[2.,1e-6],tensor=mh['tensor'],aux=mh['aux'],input_hash=mh['input_hash']);assert final.tobytes()==next_both.tobytes()
    out=d/f'{bi:06d}.profile.response.bin';second=t.command(dict(diagnostics=True,op='profile',input=str(d/f'{bi:06d}.request.bin'),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==result.tobytes()
    row=dict(label=label,layer=layer+1,tokens=n,begin=begin,success=True,kv_bitwise_equal=True,prepared_mlp_bitwise_equal=True,finished_hidden_norm_bitwise_equal=True,first_call=first,first_profile=profile,call=call,profile=second,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in [req,res,areq,ares,mreq,mres]});cases.append(row);print(json.dumps(row),flush=True)
 verify_module(t,module);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
 (d/'report.json').write_text(json.dumps(dict(module_sha256=module,q4=a.q4,front=a.front,source_hashes=hashes,cases=cases,ordinary_queries=len(t.measurements),whole_inference_reduction_verified=False),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
