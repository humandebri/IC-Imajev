#!/usr/bin/env python3
"""Profile saved ordinary full MLP requests and verify replies on same module."""
import argparse,hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--source',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);source=ROOT/a.source;sha=lambda b:hashlib.sha256(b).hexdigest();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());wasm=sha((ROOT/a.wasm).read_bytes())
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__),ROOT/'Cargo.lock'];hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths};cases=[]
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'])
try:
 verify_module(t,wasm)
 for layer in [0,1,3]:
  request=source/f'{layer:02d}-full-mlp.request.bin';expected=source/f'{layer:02d}-full-mlp.response.bin';out=d/f'{layer:02d}.response.bin';r=t.command(dict(op='profile',input=str(request),output=str(out)));assert out.read_bytes()==expected.read_bytes();cases.append(dict(layer=layer,input_sha256=sha(request.read_bytes()),output_sha256=sha(out.read_bytes()),profile=r));print(json.dumps(cases[-1]),flush=True)
 verify_module(t,wasm);assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths};result=dict(scope=__doc__,canister=a.canister,wasm_sha256=wasm,source_hashes=hashes,cases=cases,ordinary_queries=3)
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
