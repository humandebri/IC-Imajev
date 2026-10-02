#!/usr/bin/env python3
"""Wasm versus native scalar-order F32, including raw low bits and tile tails."""
import hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,encode,decode,atomic
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());h,x=decode((ROOT/'artifacts/activation-int8-fused-617/queries/000003.request.bin').read_bytes());x=x.reshape(132,2560);d=ROOT/'artifacts/linear-tiling/raw-parity';d.mkdir(exist_ok=True);t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);records=[]
try:
 for n in [1,3,4,5,7,15,16,17,31,32,33,36,63,64,65,96,127,128,129,132]:
  for rows in [1,15,16,17,33,63,64,65]:
   r={**h,'op':'matmul','tensor':'readout-f32','dims':[n,rows,2560,1],'aux':[],'scalars':[],'encoding':'bf16-exact','step':len(records)};q=d/f'{n}-{rows}.request.bin';wasm=d/f'{n}-{rows}.wasm.bin';native=d/f'{n}-{rows}.native.bin';atomic(q,encode(r,x[:n]));got=t.command(dict(op='step',input=str(q),output=str(wasm)));assert 'ok' in got,got
   subprocess.run([str(ROOT/'target/release/primitive'),str(q),str(native),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True)
   _,a=decode(wasm.read_bytes());_,b=decode(native.read_bytes());assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),(n,rows,float(abs(a-b).max()));records.append(dict(tokens=n,rows=rows,bitwise_equal=True,**got['ok']))
 report=dict(scope='Raw F32 output; lossless transport; real readout weights; nonzero row offset; 16/32/64-token and16-output-row SIMD/tails',cases=records,wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest());(ROOT/'docs/linear-tiling-parity.json').write_text(json.dumps(report,indent=2)+'\n');print('Raw F32 bit parity:',len(records),'cases')
finally:t.close()
