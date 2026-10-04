#!/usr/bin/env python3
"""Profile frozen rolled/standard requests without altering production journals."""
import argparse,hashlib,json,pathlib,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','roll','standard','directory']:ap.add_argument('--'+name,required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);assert not (d/'report.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());module=sha(ROOT/a.wasm)
paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths}
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);rows=[]
try:
 verify_module(t,module)
 for label,folder,indices in [('roll',a.roll,range(7)),('standard',a.standard,range(8))]:
  source=ROOT/folder/'queries'
  for index in indices:
   request=source/f'{index:06d}.request.bin';response=source/f'{index:06d}.response.bin';raw=request.read_bytes();h=json.loads(raw[4:4+int.from_bytes(raw[:4],'little')]);output=d/f'{label}-{index:06d}.response.bin'
   result=t.command(dict(op='profile',input=str(request),output=str(output)));assert decode(output.read_bytes())[1].tobytes()==decode(response.read_bytes())[1].tobytes()
   row=dict(label=label,index=index,op=h['op'],tensor=h.get('tensor'),request_sha256=sha(request),response_sha256=sha(response),profile=result,bitwise_equal=True);rows.append(row);print(json.dumps(row),flush=True)
 verify_module(t,module);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
 (d/'report.json').write_text(json.dumps(dict(module_sha256=module,source_hashes=hashes,profile_query_count=len(rows),production_inference_query_count=0,cases=rows),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
