#!/usr/bin/env python3
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import Transport,decode
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());d=ROOT/'artifacts/integer-tiling';d.mkdir(exist_ok=True);t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'])
try:
 start=time.perf_counter();got=t.command(dict(op='step',input=str(ROOT/'artifacts/integer-kernel/request.bin'),output=str(d/'response.bin')));elapsed=time.perf_counter()-start
 _,a=decode((d/'response.bin').read_bytes());_,b=decode((ROOT/'artifacts/integer-kernel/response.bin').read_bytes());equal=np.array_equal(a.view(np.uint32),b.view(np.uint32));assert equal
 r=dict(wall_seconds=elapsed,bitwise_equal_to_original_integer_kernel=bool(equal),before_instructions=628986326,ok=got['ok'],wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest());(ROOT/'docs/integer-tiling.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
finally:t.close()
