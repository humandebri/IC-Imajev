#!/usr/bin/env python3
import hashlib,json,pathlib,struct,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,atomic,encode
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());directory=ROOT/'artifacts/int8-wire-contracts';directory.mkdir(exist_ok=True)
t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),directory,m['pack_hash'])
h=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],input_hash='0'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='int8-block256-v1');raw=encode(h,[1.]);n=struct.unpack('<I',raw[:4])[0];offset=4+n
cases=[]
def modified(position,bytes_):
 b=bytearray(raw[:-32]);b[position:position+len(bytes_)]=bytes_;return bytes(b)+hashlib.sha256(b).digest()
try:
 for name,payload,expected in [('zero-scale',modified(offset+8,struct.pack('<f',0)),'INT8 codec scale'),('nan-scale',modified(offset+8,struct.pack('<f',float('nan'))),'INT8 codec scale'),('reserved-q',modified(offset+12,bytes([128])),'INT8 codec range'),('wrong-prefix',modified(offset+4,struct.pack('<I',0)),'INT8 codec bounds'),('over-count',modified(offset,struct.pack('<I',900001)),'INT8 codec bounds')]:
  source=directory/(name+'.request.bin');atomic(source,payload)
  try:t.command(dict(op='step',input=str(source),output=str(directory/(name+'.response.bin'))));raise AssertionError('invalid INT8 wire accepted')
  except RuntimeError as e:
   assert expected in str(e),(name,str(e));cases.append(dict(case=name,rejected=True,error=str(e)))
finally:t.close()
(ROOT/'docs/int8-wire-contracts.json').write_text(json.dumps(dict(scope='Authenticated ordinary queries on local8001, no updates',cases=cases),indent=2)+'\n');print(cases)
