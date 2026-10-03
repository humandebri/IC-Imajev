#!/usr/bin/env python3
"""Raw I32-block result and fused LoRA output versus native scalar integer oracle."""
import argparse,hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode,encode,atomic
from prefix_inference import verify_module
ap=argparse.ArgumentParser();ap.add_argument('--canister',default='46el7-ql777-77775-aaada-cai');ap.add_argument('--directory',default='artifacts/integer-parity');ap.add_argument('--output',default='docs/integer-parity.json');ap.add_argument('--token-scale',action='store_true');args=ap.parse_args();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());h,x=decode((ROOT/'artifacts/integer-kernel/request.bin').read_bytes());x=x.reshape(132,2560);x=np.concatenate([x,x[:2]]);d=ROOT/args.directory;d.mkdir(exist_ok=True);t=Transport(m['model'],'http://localhost:8001/',args.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);cases=[]
try:
 wasm=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest();verify_module(t,wasm)
 for n in [1,2,3,4,5,6,7,8,9,16,31,32,33,63,64,65,96,124,128,132,134]:
  for rows in [8,16,24]:
   for op in (['int8_matmul_token','lora_integer_token'] if args.token_scale else ['int8_matmul','lora_integer']):
    header={**h,'op':op,'dims':[n,rows,2560,8],'aux':[],'scalars':[],'step':len(cases)}
    if op in ('lora_integer','lora_integer_token'):header.update(aux=[h['tensor'].replace('.weight','.lora_A.weight'),h['tensor'].replace('.weight','.lora_B.weight')],scalars=[2.])
    q=d/f'{len(cases)}.request.bin';w=d/f'{len(cases)}.wasm.bin';native=d/f'{len(cases)}.native.bin';atomic(q,encode(header,x[:n]));got=t.command(dict(op='step',input=str(q),output=str(w)));subprocess.run([str(ROOT/'target/release/primitive'),str(q),str(native),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True)
    _,a=decode(w.read_bytes());_,b=decode(native.read_bytes());assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),(op,n,rows,float(abs(a-b).max()));cases.append(dict(op=op,tokens=n,rows=rows,bitwise_equal=True,**got['ok']))
 report=dict(activation_scale='token' if args.token_scale else 'block256',scope='Real nonzero-offset weights; raw integer dot/scaling and fused F32 LoRA; native scalar oracle; no claim of matching prior F32 arithmetic',wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),cases=cases);verify_module(t,wasm);(ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n');print('Integer raw/fused scalar parity',len(cases),'cases')
finally:t.close()
