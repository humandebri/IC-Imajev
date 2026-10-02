#!/usr/bin/env python3
"""Reject malformed wire payloads on the dedicated local canister; no updates."""
import hashlib,json,pathlib,struct,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,atomic,encode
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());d=ROOT/'artifacts/wire-contracts';d.mkdir(exist_ok=True)
t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);cases=[]
h=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],input_hash='0'*64,step=0,op='embed',tensor='model.language_model.embed_tokens.weight',dims=[1,2560],scalars=[],encoding='bf16-exact')
def raw(header,payload):
 b=json.dumps(header,separators=(',',':')).encode();v=struct.pack('<I',len(b))+b+payload;return v+hashlib.sha256(v).digest()
valid=encode(h,[1.]);bad=bytearray(valid);bad[-1]^=1
inputs=[('checksum',bytes(bad),'checksum'),('unknown-codec',raw({**h,'encoding':'invalid'},b''),'unsupported encoding'),('over-count',raw(h,struct.pack('<I',900001)),'codec count'),('padding',raw(h,struct.pack('<I',1)+bytes([128])+bytes([0,0])),'codec padding'),('trailing',raw(h,struct.pack('<I',1)+bytes([0])+bytes([0,0,0])),'codec trailing'),('wrong-pack',encode({**h,'pack_hash':'0'*64},[1.]),'model mismatch')]
try:
 for name,request,expected in inputs:
  source=d/(name+'.request.bin');atomic(source,request)
  try:t.command(dict(op='step',input=str(source),output=str(d/(name+'.response.bin'))));raise AssertionError('invalid state accepted')
  except RuntimeError as e:
   assert expected in str(e),(name,str(e));cases.append(dict(case=name,rejected=True,error=str(e)))
finally:t.close()
(ROOT/'docs/wire-contracts.json').write_text(json.dumps(dict(scope='Authenticated ordinary queries on dedicated local8001; no updates',cases=cases),indent=2)+'\n');print(json.dumps(cases,indent=2))
