#!/usr/bin/env python3
"""Replay every actual attention-head query from the unchanged baseline graph."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode
src=ROOT/'artifacts/linear-tiling-full-617';dest=ROOT/'artifacts/exact-optimization/attention-check';dest.mkdir(parents=True,exist_ok=True);r=json.loads((src/'first-report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash']);cases=[]
try:
 for q in r['queries']:
  if q['op']!='attention_heads_bf16':continue
  i=q['index'];reply=dest/f'{i}.response.bin';got=t.command(dict(op='step',input=str(src/'queries'/f'{i:06d}.request.bin'),output=str(reply)));assert 'ok' in got,got;_,a=decode(reply.read_bytes());_,b=decode((src/'queries'/f'{i:06d}.response.bin').read_bytes());assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),i;cases.append(dict(index=i,bitwise_equal=True,before_instructions=q['ok']['instructions'],**got['ok']))
 report=dict(cases=cases,before_total=sum(q['before_instructions'] for q in cases),after_total=sum(q['instructions'] for q in cases),wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest());(ROOT/'docs/attention-exact-check.json').write_text(json.dumps(report,indent=2)+'\n');print('attention exact:',len(cases),report['before_total'],report['after_total'])
finally:t.close()
