#!/usr/bin/env python3
"""Compare raw partial-finish query to saved exact compressed-finish output."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());module=sha(ROOT/a.wasm)
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths}
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='mlp-down-state-exact-v1',frame_version=3);cases=[]
try:
 verify_module(t,module)
 for label,folder in [('617','pair-down1600-check-617-v2'),('maximum','pair-down1280-check-maximum-v1')]:
  source=ROOT/'artifacts/f32_k_continue'/folder;r=json.loads((source/'report.json').read_text());c=r['cases'][0];n=c['tokens'];rows=c['partial_down_rows'];index=c['follow_call']['index'];request=source/f'{index+2:06d}.request.bin';response=source/f'{index+2:06d}.response.bin'
  raw=request.read_bytes();h=json.loads(raw[4:4+int.from_bytes(raw[:4],'little')]);_,state=decode((source/f'{index:06d}.response.bin').read_bytes());_,expected=decode(response.read_bytes());state=state[:n*(2560+9216+100)]
  qi=t.index;y=t.run('mlp_down_norm_partial_prepared',state,[n,2560,rows],[2.,1e-6],tensor=h['tensor'],aux=h['aux'],input_hash=h['input_hash']);assert y.tobytes()==expected.tobytes();call=t.measurements[-1]
  out=d/f'{qi:06d}.profile.response.bin';profile=t.command(dict(op='profile',input=str(d/f'{qi:06d}.request.bin'),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==y.tobytes();spans={name:cost for name,cost,count in profile['ok']['spans']};assert 'carry_inflate'not in spans
  cases.append(dict(label=label,tokens=n,down_rows=rows,call=call,profile=profile,source_request_sha256=sha(request),source_response_sha256=sha(response),source_report_sha256=sha(source/'report.json'),bitwise_equal=True));print(json.dumps(cases[-1]),flush=True)
 verify_module(t,module);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
 (d/'report.json').write_text(json.dumps(dict(module_sha256=module,cases=cases,source_hashes=hashes,ordinary_queries=len(t.measurements),profile_queries=len(cases),whole_inference_reduction_verified=False),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
