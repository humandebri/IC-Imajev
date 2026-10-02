#!/usr/bin/env python3
import hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode,encode,atomic
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());src=next((ROOT/'artifacts/exact-optimization/grouped-check').glob('*.request.bin'));h,x=decode(src.read_bytes());d=ROOT/'artifacts/exact-optimization/contracts';d.mkdir(exist_ok=True);t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);records=[]
try:
 for name,dims in [('zero-width',[132,0,2560,0,1236]),('more-than-three',[132,1236,2560,0,3709]),('wrong-cols',[132,1236,2559,0,2472]),('row-outside',[132,1236,2560,262144,2472]),('work-limit',[132,1400,2560,0,4200]),('zero-total',[132,1236,2560,0,0])]:
  header={**h,'dims':dims};request=d/(name+'.request.bin');atomic(request,encode(header,x))
  try:t.command(dict(op='step',input=str(request),output=str(d/(name+'.response.bin'))));raise AssertionError('malformed grouped query accepted')
  except RuntimeError as e:records.append(dict(case=name,rejected=True,error=str(e)))
 (ROOT/'docs/grouped-contracts.json').write_text(json.dumps(dict(cases=records),indent=2)+'\n');print('Grouped invalid contracts rejected',len(records))
finally:t.close()
