#!/usr/bin/env python3
"""Real MLP product and original F32 A, exact client-held column continuation.

Diagnostic calls count separately. Does not claim full-model query reduction.
"""
import argparse,hashlib,json,pathlib,re,subprocess,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--layers',default='0,1,3');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest();source=ROOT/'artifacts/prefix_codec/full-s1-wide-proof/617';raw=(source/'report.json').read_bytes();r=json.loads(raw);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());helper=ROOT/'artifacts/f32-block-native/release/f32_args';wasm=sha((ROOT/a.wasm).read_bytes());cases=[]
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__),ROOT/'Cargo.lock',ROOT/'MODEL_LOCK.json',ROOT/'checkpoints/full-int8.manifest.json',ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'canisters/inference/Cargo.toml'];hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='bf16-block256-exact-v1',frame_version=3)
try:
 verify_module(t,wasm)
 for layer in map(int,a.layers.split(',')):
  q=next(q for q in r['queries']if q['op']=='mlp_full_integer'and f'.layers.{layer}.'in q['tensor']);ip=source/'queries'/f"{q['index']:06d}.request.bin";h,x=decode(ip.read_bytes());n=h['dims'][0];root=f'model.language_model.layers.{layer}';norm=t.run('add_norm_bf16',x,[n,2560],[1e-6],tensor=h['tensor'],input_hash=h['input_hash']);assert norm.size==n*5120
  product=t.run('mlp_gate_up_integer',norm[n*2560:],[n,9216,2560,0],[2.],tensor=root+'.mlp.gate_proj.weight',aux=[root+'.mlp.up_proj.weight'],input_hash=h['input_hash']).reshape(n,9216);np.save(d/f'{layer:02d}-product.npy',product)
  tensor=root+'.mlp.down_proj.lora_A.weight';weight=next(x for x in m['tensors']if x['name']==tensor);assert(weight['rows'],weight['cols'],weight['dtype'])==(64,9216,'f32');wp=d/f'{layer:02d}-weight.bin'
  with (ROOT/'checkpoints/full-int8.pack').open('rb')as f:f.seek(weight['offset']);wb=f.read(weight['bytes'])
  wp.write_bytes(wb)
  for tokens in ([1,7,n]if layer==0 else[n]):
   full=product[:tokens];p=d/f'{layer:02d}-{tokens}-input.bin';p.write_bytes(full.astype('<f4').tobytes());native=json.loads(subprocess.check_output([str(helper),'native',str(tokens),'64','9216',str(wp),str(p)],text=True,cwd=ROOT));expected=t.run('matmul',full.ravel(),[tokens,64,9216],tensor=tensor,input_hash=h['input_hash']);assert list(hashlib.sha256(expected.astype('<f4').tobytes()).digest())==native['digest'];results=[]
   for chunk in [256,1024,3072]:
    acc=np.zeros(tokens*64,np.float32);start=len(t.measurements);max_state=0
    for begin in range(0,9216,chunk):
     count=min(chunk,9216-begin);state=np.concatenate([full[:,begin:begin+count].ravel(),acc]);max_state=max(max_state,state.nbytes);acc=t.run('matmul_k_continue',state,[tokens,64,9216,begin,count],tensor=tensor,input_hash=h['input_hash'])
    assert acc.tobytes()==expected.tobytes();calls=t.measurements[start:];results.append(dict(chunk=chunk,ordinary_queries=len(calls),instructions=sum(q['ok']['instructions']for q in calls),candid_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes']for q in calls),maximum_raw_carry_bytes=max_state,maximum_query_instructions=max(q['ok']['instructions']for q in calls),bitwise_equal=True,output_sha256=sha(acc.tobytes())))
   row=dict(layer=layer,tokens=tokens,source_request_sha256=sha(ip.read_bytes()),product_sha256=sha(full.tobytes()),weight_sha256=sha(wb),native=native,full_output_sha256=sha(expected.tobytes()),continuations=results);cases.append(row);print(json.dumps(row),flush=True)
 verify_module(t,wasm);assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths};report=dict(scope=__doc__,canister=a.canister,wasm_sha256=wasm,source_hashes=hashes,helper_sha256=sha(helper.read_bytes()),source_report_sha256=sha(raw),model=m['model'],pack_hash=m['pack_hash'],ordinary_queries=len(t.measurements),cases=cases,goal_50_verified=False)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
