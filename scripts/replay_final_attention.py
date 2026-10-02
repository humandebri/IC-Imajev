#!/usr/bin/env python3
"""Final signed-zero fix replay for every attention query of both completed full graphs."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());d=ROOT/'artifacts/exact-optimization/final-replay';d.mkdir(exist_ok=True);t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);records=[]
try:
 for name in ['exact-grouped-full-617','exact-grouped-lossless-617']:
  src=ROOT/'artifacts'/name;r=json.loads((src/'first-report.json').read_text());cases=[]
  for q in r['queries']:
   if q['op']!='attention_heads_bf16':continue
   i=q['index'];p=d/f'{name}-{i}.bin';got=t.command(dict(op='step',input=str(src/'queries'/f'{i:06d}.request.bin'),output=str(p)));assert 'ok' in got,got;_,a=decode(p.read_bytes());_,b=decode((src/'queries'/f'{i:06d}.response.bin').read_bytes());assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),(name,i);cases.append(dict(index=i,bitwise_equal=True,before_instructions=q['ok']['instructions'],after_instructions=got['ok']['instructions']))
  records.append(dict(source=name,full_run_wasm=r['wasm_sha256'],raw_report_sha256=hashlib.sha256((src/'first-report.json').read_bytes()).hexdigest(),cases=cases))
 (ROOT/'docs/exact-final-wasm-replay.json').write_text(json.dumps(dict(final_wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),runs=records,scope='Signed-zero correction after full runs; all actual queries of the changed operation replayed with identical outputs; full-run timing belongs to the prior module'),indent=2)+'\n');print('Final module matches all',sum(len(r['cases']) for r in records),'actual Attention queries')
finally:t.close()
