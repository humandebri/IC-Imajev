#!/usr/bin/env python3
"""Compare v2 server checksum with v3 host checksum on identical real MLP inputs."""
import argparse,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode,atomic
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest();rows=[]
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'])
try:
 verify_module(t,sha((ROOT/a.wasm).read_bytes()))
 for layer in [0,1,3]:
  source=ROOT/'artifacts/quantize_scan/mlp-frames'/f'{layer:02d}-full-mlp.request.bin';h,x=decode(source.read_bytes());expected=decode(source.with_name(f'{layer:02d}-full-mlp.response.bin').read_bytes())[1];profiles={}
  for version in [2,3]:
   req=d/f'{layer:02d}-v{version}.request.bin';out=d/f'{layer:02d}-v{version}.response.bin';atomic(req,encode(dict(h,version=version),x));r=t.command(dict(op='profile',input=str(req),output=str(out)));rh,y=decode(out.read_bytes());assert rh==dict(h,version=version,step=h['step']+1) and y.tobytes()==expected.tobytes();profiles[str(version)]=r
  rows.append(dict(layer=layer,bitwise_equal=True,profiles=profiles));print(json.dumps(rows[-1]),flush=True)
 verify_module(t,sha((ROOT/a.wasm).read_bytes()))
finally:t.close()
(d/'report.json').write_text(json.dumps(dict(canister=a.canister,wasm_sha256=sha((ROOT/a.wasm).read_bytes()),cases=rows),indent=2)+'\n')
