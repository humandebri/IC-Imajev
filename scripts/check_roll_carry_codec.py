#!/usr/bin/env python3
"""Compare existing Huffman and DEFLATE carry codecs on actual rolled states."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
from mlp_delta_carry import HUFFMAN_NAME,encode_request
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','roll','prefix','directory']:ap.add_argument('--'+name,required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);assert not (d/'report.json').exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());module=sha(ROOT/a.wasm)
paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths}
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);cases=[]
try:
 verify_module(t,module)
 for label in ['617','insufficient']:
  root=ROOT/a.roll/label;report=json.loads((root/'report.json').read_text())
  for q in report['queries']:
   if q['op']!='mlp_finish_delta_log_integer':continue
   index=q['index'];req=root/'queries'/f'{index:06d}.request.bin';res=root/'queries'/f'{index:06d}.response.bin';raw=req.read_bytes();h=json.loads(raw[4:4+int.from_bytes(raw[:4],'little')]);n,p,rows=h['dims'];layer=int(h['tensor'].split('.')[3]);_,state=decode((root/'queries'/f'{index-1:06d}.response.bin').read_bytes());_,expected=decode(res.read_bytes())
   with np.load(ROOT/a.prefix/f'states/layer-{layer+1:02d}.npz',allow_pickle=False)as z:cv=z['conv'].copy();log=z['delta_log'].copy()
   h.update(step=t.index,encoding=HUFFMAN_NAME);packet=encode_request(h,state[:n*11876],cv,log);query_index=t.index;y=t._run_encoded(h,packet);assert y.tobytes()==expected.tobytes();call=t.measurements[-1]
   output=d/f'{query_index:06d}.profile.response.bin';profile=t.command(dict(op='profile',input=str(d/f'{query_index:06d}.request.bin'),output=str(output)));assert decode(output.read_bytes())[1].tobytes()==y.tobytes()
   row=dict(label=label,layer=layer,tokens=n,down_rows=rows,original_call=q,call=call,profile=profile,request_sha256=sha(req),response_sha256=sha(res),bitwise_equal=True);cases.append(row);print(json.dumps(row),flush=True)
 verify_module(t,module);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
 (d/'report.json').write_text(json.dumps(dict(module_sha256=module,source_hashes=hashes,ordinary_queries=len(t.measurements),profile_queries=len(cases),whole_inference_reduction_verified=False,cases=cases),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
