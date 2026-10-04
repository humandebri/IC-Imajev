#!/usr/bin/env python3
"""Compare shared GQA views against unchanged individual attention queries in Wasm."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
if not __debug__:raise RuntimeError('Proof checker requires assertions')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
if any(d.iterdir()):raise ValueError('Use a new evidence directory')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','scripts/build_full_prefix_candidate.py']]+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};module=sha(ROOT/a.wasm);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());rows=[]
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='bf16-block256-exact-v1',frame_version=3)
try:
 verify_module(t,module)
 for seed,(n,width,heads,prefix)in enumerate([(1,1,1,0),(3,3,2,5),(7,8,4,3),(45,256,4,0),(80,256,4,45),(87,256,16,45),(89,256,16,45),(1,256,16,131)]):
  count=(heads*n+2*((heads+3)//4)*(n+prefix))*width;rng=np.random.default_rng(seed);x=rng.normal(0,.5,count).astype('<f4');x=(x.view('<u4')&0xffff0000).view('<f4');x[::127]=-0.;np.save(d/f'fixture-{seed}.npy',x);identity=hashlib.sha256(x.tobytes()).hexdigest()
  stride=n*width;kvstride=(n+prefix)*width;groups=(heads+3)//4;expected=[];single=[]
  for head in range(heads):
   group=head//4;parts=[x[head*stride:(head+1)*stride],x[heads*stride+group*kvstride:heads*stride+(group+1)*kvstride],x[heads*stride+(groups+group)*kvstride:heads*stride+(groups+group+1)*kvstride]]
   expected.append(t.run('attention_suffix_bf16',np.concatenate(parts),[n,width,prefix],[],input_hash=identity));single.append(t.measurements[-1])
  got=t.run('gqa_suffix_bf16',x,[n,width,heads,prefix],[],input_hash=identity);metric=t.measurements[-1]
  assert got.tobytes()==np.concatenate(expected).tobytes()
  row=dict(n=n,width=width,heads=heads,prefix=prefix,fixture_sha256=identity,bitwise_equal=True,shared=metric,individual=single);rows.append(row);print(json.dumps(row),flush=True)
 verify_module(t,module)
 assert sha(ROOT/a.wasm)==module and hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
 (d/'report.json').write_text(json.dumps(dict(module_sha256=module,source_hashes=hashes,cases=rows,production_query_reduction_verified=False),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
