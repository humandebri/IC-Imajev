#!/usr/bin/env python3
"""Wire-size exploration only: lossless shuffle+zlib is NOT a canister codec yet."""
import hashlib,json,pathlib,struct,zlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'artifacts/optimization-check/projection').glob('*.bin')):
 raw=p.read_bytes();header=struct.unpack('<I',raw[:4])[0];start=4+header
 payload=raw[start:-32];n=len(payload)//4
 prepared=raw[:start]+b''.join(payload[lane::4] for lane in range(4))+raw[-32:]
 packed=zlib.compress(prepared,1)
 decoded=zlib.decompress(packed);restored=bytearray(len(payload))
 for lane in range(4):restored[lane::4]=decoded[start+lane*n:start+(lane+1)*n]
 assert decoded[:start]+restored+decoded[-32:]==raw
 metadata=json.loads(raw[4:start]);rows.append(dict(file=p.name,op=metadata['op'],dims=metadata['dims'],direction=p.name.split('.')[-2],raw_bytes=len(raw),envelope_plus_zlib_bytes=min(len(raw),len(packed)+16),sha256=hashlib.sha256(raw).hexdigest()))
report=dict(scope='Host-only size estimate for a 16-byte envelope; no measured query compression instructions/time. No quantization.',codec='float byte-plane shuffle + zlib level1',records=rows)
(ROOT/'artifacts/optimization-check/transport-exploration.json').write_text(json.dumps(report,indent=2)+'\n')
for op in ['matmul_reference','matmul','lora_project','lora_finish']:
 qs=[x for x in rows if x['op']==op and (x['dims'][0]==132 or x['dims'][0]==132*256)]
 print(op,sum(x['raw_bytes'] for x in qs),sum(x['envelope_plus_zlib_bytes'] for x in qs))
