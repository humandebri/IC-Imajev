#!/usr/bin/env python3
"""Compare RoPE queries before and after fixed-angle update preparation."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,encode,atomic
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--phase',choices=['baseline','candidate'],required=True);ap.add_argument('--directory',default='artifacts/prepared-rope/probe');a=ap.parse_args()
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest();t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d/a.phase,m['pack_hash']);cases=[]
 try:
  verify_module(t,sha)
  if a.phase=='candidate':
   first=next(t for t in m['tensors'] if t['dtype'] in ('int8','f32') and 'embed_tokens' not in t['name'])
   preparation=t.command(dict(op='warm_weights',name=first['name']))
   assert preparation['ok']['cache']['rope_bytes']==131072
   before=t.command(dict(op='weight_cache_status'))['ok']['cache']
   (d/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
  for label in ['prefix','617','insufficient','maximum','normal']:
   source=ROOT/f'artifacts/blake3-v1-{label}';r=json.loads((source/'report.json').read_text())
   for part in ['q','k']:
    q=next(q for q in r['queries'] if q['op']=='norm_rope_heads_bf16' and q['tensor']==f'model.language_model.layers.3.self_attn.{part}_norm.weight');path=source/'queries'/f'{q["index"]:06d}.request.bin';h,v=decode(path.read_bytes());h.update(op='rope_heads',tensor='',aux=[],scalars=[h['scalars'][1]])
    configurations=[('fixed',h)]
    if label=='617' and part=='q':
     configurations += [('position-fallback',dict(h,dims=[*h['dims'][:3],512,h['dims'][4]])),('theta-fallback',dict(h,scalars=[500000.])),('rotary-fallback',dict(h,dims=[h['dims'][0],256,8,*h['dims'][3:]]))]
    for kind,request_h in configurations:
     name=f'{label}-{part}-{kind}';request=d/f'{name}.request.bin';reply=d/f'{name}.{a.phase}.response.bin';encoded=encode(request_h,v)
     if a.phase=='baseline':atomic(request,encoded)
     else:assert request.read_bytes()==encoded
     measured=t.command(dict(op='step',input=str(request),output=str(reply)));_,out=decode(reply.read_bytes());case=dict(name=name,dims=request_h['dims'],source_request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),request_sha256=hashlib.sha256(encoded).hexdigest(),response_sha256=hashlib.sha256(reply.read_bytes()).hexdigest(),measurement=measured)
     if a.phase=='candidate':
      _,expected=decode((d/f'{name}.baseline.response.bin').read_bytes());assert np.array_equal(out.view(np.uint32),expected.view(np.uint32));case['bitwise_equal']=True
     cases.append(case);print(json.dumps(case),flush=True)
  if a.phase=='candidate':assert before==t.command(dict(op='weight_cache_status'))['ok']['cache']
  verify_module(t,sha)
  sources=[pathlib.Path(__file__),ROOT/'crates/imajev-runtime/src/rope.rs',ROOT/'crates/imajev-runtime/src/lib.rs',ROOT/'canisters/inference/src/lib.rs']
  (d/f'{a.phase}.report.json').write_text(json.dumps(dict(scope='RoPE-only handlers on saved projection inputs, not full-model accuracy/speed/query count; unsupported parameters use original computation',wasm_sha256=sha,canister=a.canister,phase=a.phase,cases=cases,ordinary_queries=len(cases),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}),indent=2)+'\n')
 finally:t.close()
if __name__=='__main__':main()
