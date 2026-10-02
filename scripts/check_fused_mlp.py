#!/usr/bin/env python3
"""Actual weights: fused gate/up/SwiGLU versus three existing ordinary queries."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode,encode,atomic
ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);args=ap.parse_args();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());source=ROOT/'artifacts/bottleneck-fused-full-617/queries';r=json.loads((source.parent/'first-report.json').read_text())
record=next(q for q in r['queries'] if q['op']=='lora_integer' and '.mlp.gate_proj.weight' in q['tensor']);header,x=decode((source/f"{record['index']:06d}.request.bin").read_bytes());x=x.reshape(132,2560);dest=ROOT/'artifacts/fused-mlp-check';dest.mkdir(exist_ok=True);t=Transport(m['model'],'http://localhost:8001/',args.canister,str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash'],'bf16-exact');cases=[]
try:
 for n in [1,7,8,9,32,36,64,96,132]:
  for rows,start in [(8,8),(24,0),(1024,4096)]:
   gate=header['tensor'];up=gate.replace('.gate_proj.weight','.up_proj.weight')
   a=t.run('lora_integer',x[:n],[n,rows,2560,start],[2.],tensor=gate,aux=[gate.replace('.weight','.lora_A.weight'),gate.replace('.weight','.lora_B.weight')])
   b=t.run('lora_integer',x[:n],[n,rows,2560,start],[2.],tensor=up,aux=[up.replace('.weight','.lora_A.weight'),up.replace('.weight','.lora_B.weight')])
   expected=t.run('swiglu_bf16',np.concatenate([a,b]),[n*rows]);before=sum(q['ok']['instructions'] for q in t.measurements[-3:])
   actual=t.run('mlp_gate_up_integer',x[:n],[n,rows,2560,start],[2.],tensor=gate,aux=[up]);assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),(n,rows,start)
   cases.append(dict(tokens=n,rows=rows,start=start,bitwise_equal=True,separate_instructions=before,fused_instructions=t.measurements[-1]['ok']['instructions']))
 # A full scheduler-sized fused query, and metadata/work rejection before unsafe loads.
 for rows,start in [(4096,0),(4096,4096),(1024,8192)]:
  out=t.run('mlp_gate_up_integer',x,[132,rows,2560,start],[2.],tensor=gate,aux=[up]);cases.append(dict(tokens=132,rows=rows,start=start,instructions=t.measurements[-1]['ok']['instructions'],scope='Scheduler shape; full graph verifies bit parity'))
 invalid=[]
 for name,patch in [('wrong-up',{'aux':[up.replace('layers.0','layers.1')]}),('missing-scale',{'scalars':[]}),('rows-not-eight',{'dims':[132,9,2560,0]}),('outside-row',{'dims':[132,8,2560,9216]}),('too-much-work',{'dims':[132,6816,2560,0]})]:
  h={**header,'op':'mlp_gate_up_integer','dims':[132,16,2560,0],'scalars':[2.],'aux':[up],**patch};p=dest/f'{name}.request.bin';atomic(p,encode(h,x))
  try:t.command(dict(op='step',input=str(p),output=str(dest/f'{name}.response.bin')));raise AssertionError(name)
  except RuntimeError as e:invalid.append(dict(case=name,rejected=True,error=str(e)))
 report=dict(canister=args.canister,wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),cases=cases,invalid_cases=invalid);(ROOT/'docs/fused-mlp-check.json').write_text(json.dumps(report,indent=2)+'\n');print('Fused MLP',len(cases),'cases; rejected',len(invalid),flush=True)
finally:t.close()
