#!/usr/bin/env python3
"""Raw F32 optimized/scalar Wasm parity: signed zero, value lanes, widths/tokens and BF16."""
import hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,encode,decode,atomic
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());d=ROOT/'artifacts/exact-optimization/attention-boundaries';d.mkdir(parents=True,exist_ok=True);t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);cases=[]
try:
 for n in [1,3,4,7,16]:
  for width in [3,4,5,8,16]:
   for zero in [False,True]:
    h=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],input_hash='0'*64,step=len(cases),op='attention_bf16' if n%2 else 'attention',tensor='',dims=[n,width],scalars=[],encoding='bf16-exact');v=np.sin(np.arange(n*width*3,dtype=np.float32)*np.float32(.137)).astype(np.float32)
    if zero:v[:2*n*width]=0.;v[2*n*width:]=-0.
    i=len(cases);request=d/f'{i}.request.bin';wasm=d/f'{i}.wasm.bin';native=d/f'{i}.native.bin';atomic(request,encode(h,v));got=t.command(dict(op='step',input=str(request),output=str(wasm)));assert 'ok' in got,got;ref=d/f'{i}.reference.request.bin';atomic(ref,encode({**h,'op':h['op']+'_reference'},v));reference=t.command(dict(op='step',input=str(ref),output=str(native)));assert 'ok' in reference,reference;_,a=decode(wasm.read_bytes());_,b=decode(native.read_bytes());assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),(n,width,zero,a.view(np.uint32),b.view(np.uint32));cases.append(dict(tokens=n,width=width,signed_zero_case=zero,bitwise_equal=True))
 (ROOT/'docs/attention-boundaries.json').write_text(json.dumps(dict(cases=cases,wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()),indent=2)+'\n');print('Attention raw F32 same-Wasm parity',len(cases),'cases')
finally:t.close()
