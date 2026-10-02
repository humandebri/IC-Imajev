#!/usr/bin/env python3
"""Wasm SIMD quantization/scaling against scalar: ties, signed zero, subnormals, range."""
import argparse,hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode,encode,atomic
ap=argparse.ArgumentParser();ap.add_argument('--canister',default='46el7-ql777-77775-aaada-cai');args=ap.parse_args();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());h,_=decode((ROOT/'artifacts/integer-kernel/request.bin').read_bytes());d=ROOT/'artifacts/bottleneck/quantization-edges';d.mkdir(parents=True,exist_ok=True)
patterns={'zero':np.array([0.,-0.],dtype=np.float32),'ties':np.array([127.,-127.,.5,-.5,1.5,-1.5,2.5,-2.5,126.5,-126.5],dtype=np.float32),'subnormals':np.array([1,0x80000001,2,0x80000002,0x007fffff,0x807fffff],dtype=np.uint32).view(np.float32),'large':np.array([1e30,-1e30,1e29,-1e29],dtype=np.float32)}
t=Transport(m['model'],'http://localhost:8001/',args.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);cases=[]
try:
 for name,v in patterns.items():
  for n in [1,9]:
   x=np.tile(np.resize(v,256),(n,10))
   for op in ['int8_matmul','lora_integer']:
    header={**h,'op':op,'dims':[n,8,2560,8],'aux':[],'scalars':[],'step':0,'encoding':'bf16-exact'}
    if op=='lora_integer':header.update(aux=[h['tensor'].replace('.weight','.lora_A.weight'),h['tensor'].replace('.weight','.lora_B.weight')],scalars=[2.])
    i=len(cases);q=d/f'{i}.request.bin';w=d/f'{i}.wasm.bin';native=d/f'{i}.native.bin';atomic(q,encode(header,x));got=t.command(dict(op='step',input=str(q),output=str(w)));assert 'ok' in got,got
    subprocess.run([str(ROOT/'target/release/primitive'),str(q),str(native),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True)
    _,a=decode(w.read_bytes());_,b=decode(native.read_bytes());assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),(name,n,op,a,b)
    cases.append(dict(pattern=name,tokens=n,op=op,bitwise_equal=True,**got['ok']))
 out=dict(wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),cases=cases);(ROOT/'docs/simd-quantization-edges.json').write_text(json.dumps(out,indent=2)+'\n');print('SIMD edge cases',len(cases),'bitwise equal')
finally:t.close()
