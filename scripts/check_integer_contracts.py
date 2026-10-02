#!/usr/bin/env python3
"""Reject malformed integer fused requests before unsafe arithmetic."""
import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode,encode,atomic
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());h,x=decode((ROOT/'artifacts/integer-full-617/queries/000003.request.bin').read_bytes());d=ROOT/'artifacts/integer-contracts';d.mkdir(exist_ok=True);t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);cases=[]
try:
 for name,patch in [('rows-not-eight',{'dims':[132,9,2560,0]}),('bad-cols',{'dims':[132,16,2559,0]}),('missing-lora',{'aux':[]}),('missing-scale',{'scalars':[]}),('outside-row',{'dims':[132,16,2560,8192]}),('wrong-a',{'aux':[h['aux'][1],h['aux'][1]]}),('too-much-work',{'dims':[132,8192,2560,0]})]:
  header={**h,**patch};q=d/(name+'.request.bin');atomic(q,encode(header,x))
  try:t.command(dict(op='step',input=str(q),output=str(d/(name+'.response.bin'))));raise AssertionError('invalid integer request accepted')
  except RuntimeError as e:cases.append(dict(case=name,rejected=True,error=str(e)))
 (ROOT/'docs/integer-contracts.json').write_text(json.dumps(dict(cases=cases),indent=2)+'\n');print('Invalid integer contracts rejected',len(cases))
finally:t.close()
