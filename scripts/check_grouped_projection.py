#!/usr/bin/env python3
"""Actual two-tile queries preserve original INT8 boundaries, including partial blocks."""
import hashlib,json,pathlib,struct,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode,atomic
src=ROOT/'artifacts/linear-tiling-full-617';d=ROOT/'artifacts/exact-optimization/grouped-check';d.mkdir(parents=True,exist_ok=True);r=json.loads((src/'first-report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());qs=r['queries'];t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);cases=[];seen=set()
try:
 for a,b in zip(qs,qs[1:]):
  if a['op']!='lora_project' or b['op']!='lora_project' or a['tensor']!=b['tensor']:continue
  p=src/'queries'/f"{a['index']:06d}.request.bin";raw=p.read_bytes();h,x=decode(raw);hb,xb=decode((src/'queries'/f"{b['index']:06d}.request.bin").read_bytes());n,w,c,start=h['dims'];n2,w2,c2,start2=hb['dims'];key=(n,w,c,w2)
  if key in seen or n2!=n or c2!=c or start2!=start+w or w2>w or not np.array_equal(x.view(np.uint32),xb.view(np.uint32)):continue
  seen.add(key);h['op']='lora_grouped';h['dims']=[n,w,c,start,w+w2];header=json.dumps(h,separators=(',',':')).encode();offset=4+struct.unpack('<I',raw[:4])[0];body=struct.pack('<I',len(header))+header+raw[offset:-32];request=d/f'{len(cases)}.request.bin';response=d/f'{len(cases)}.response.bin';atomic(request,body+hashlib.sha256(body).digest());got=t.command(dict(op='step',input=str(request),output=str(response)));assert 'ok' in got,got;_,y=decode(response.read_bytes())
  cursor=0
  for q,count in [(a,w),(b,w2)]:
   _,old=decode((src/'queries'/f"{q['index']:06d}.response.bin").read_bytes());length=n*count;assert np.array_equal(y[cursor:cursor+length].view(np.uint32),old.view(np.uint32)),key;cursor+=((length+255)//256)*256
  record=dict(dims=h['dims'],bitwise_equal=True,before_instructions=a['ok']['instructions']+b['ok']['instructions'],before_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in [a,b]),**got['ok']);cases.append(record);print(record,flush=True)
  if len(cases)>=10:break
 (ROOT/'docs/grouped-projection-check.json').write_text(json.dumps(dict(cases=cases,wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()),indent=2)+'\n')
finally:t.close()
