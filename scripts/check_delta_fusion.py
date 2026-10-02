#!/usr/bin/env python3
"""Replay actual single-head query inputs through fused native and Wasm kernels."""
import hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode,atomic
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());source=ROOT/'artifacts/activation-int8-617/queries';dest=ROOT/'artifacts/delta-fusion';dest.mkdir(exist_ok=True)
metrics=[json.loads(p.read_text()) for p in sorted(source.glob('*.metric.json'))];selected=[q for q in metrics if q['op']=='delta_bf16'][:32];assert len(selected)==32
headers=[];values=[];expected=[]
for q in selected:
 i=q['index'];h,x=decode((source/f'{i:06d}.request.bin').read_bytes());_,y=decode((source/f'{i:06d}.response.bin').read_bytes());headers.append(h);values.append(x);expected.append(y)
n,k,v=headers[0]['dims'];a=n*(2*k+v);tail=2*n+k*v
transport=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash'],wire_codec='int8-block256-v1')
records=[]
try:
 for heads in [1,8,16]:
  h={**headers[0],'step':heads,'op':'delta_heads_bf16','dims':[n,k,v,heads]};payload=encode(h,np.concatenate([x[:a] for x in values[:heads]]+[x[a:] for x in values[:heads]]));request=dest/f'heads-{heads}.request.bin';response=dest/f'heads-{heads}.response.bin';native=dest/f'heads-{heads}.native.bin';atomic(request,payload)
  got=transport.command(dict(op='step',input=str(request),output=str(response)))
  subprocess.run([str(ROOT/'target/release/primitive'),str(request),str(native),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True)
  _,wasm=decode(response.read_bytes());_,cpu=decode(native.read_bytes());old=np.concatenate([x[:n*v] for x in expected[:heads]]+[x[n*v:] for x in expected[:heads]])
  assert np.array_equal(wasm.view(np.uint32),cpu.view(np.uint32)),float(abs(wasm-cpu).max())
  record=dict(heads=heads,native_wasm_bitwise_equal=True,old_single_head_bitwise_equal=bool(np.array_equal(wasm.view(np.uint32),old.view(np.uint32))),old_single_head_max_error=float(abs(wasm-old).max()),old_single_head_instruction_sum=sum(q['ok']['instructions'] for q in selected[:heads]),**got['ok']);records.append(record);print(record,flush=True)
 (ROOT/'docs/delta-fusion-check.json').write_text(json.dumps(dict(wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),cases=records),indent=2)+'\n')
finally:transport.close()
