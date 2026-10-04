#!/usr/bin/env python3
"""Measure 2-to-1 MLP/Attention fusion against frozen canister-produced tensors."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
if not __debug__:raise RuntimeError('Proof checker requires assertions')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_bytes,read_report,selections
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
from mlp_attention_finish_codec import NAME,C,H,encode_request
from mlp_codec import NAME as DOWN
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--cases',default='617,insufficient,maximum');ap.add_argument('--layers',default='6,14,22');ap.add_argument('--rows',default='768,1280');a=ap.parse_args()
labels=selections(a.cases,('617','insufficient','maximum'),'cases');layers=list(map(int,selections(a.layers,tuple(map(str,range(2,30,4))),'layers')));widths=list(map(int,selections(a.rows,tuple(map(str,range(32,C,32))),'rows')))
d=ROOT/a.directory
if (d/'report.json').exists():raise ValueError('Use a new evidence directory')
d.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();refs={};records=[]
paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','scripts/proof_inputs.py','scripts/build_full_prefix_candidate.py']]+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths}
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());module=sha(ROOT/a.wasm);base=ROOT/'artifacts/prefix_codec/full-attention-q4-proof-v1'
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],frame_version=3)
def reference(label,op,layer):
 root=base/label;r=read_report(root/'report.json',ROOT,refs);q=next(q for q in r['queries']if q['op']==op and f'.layers.{layer}.'in q['tensor']);prefix=root/'queries'/f"{q['index']:06d}";h,x=decode(read_bytes(prefix.with_suffix('.request.bin'),ROOT,refs));_,y=decode(read_bytes(prefix.with_suffix('.response.bin'),ROOT,refs));return h,x,y
try:
 verify_module(t,module)
 for label in labels:
  for layer in layers:
   mh,x,expected_mlp=reference(label,'mlp_full_integer',layer);ah,ax,expected_attention=reference(label,'attention_full_integer',layer+1);n=mh['dims'][0];p=ah['dims'][1];assert ax[:n*C].tobytes()==expected_mlp[n*C:].tobytes()
   t.wire_codec='bf16-block256-exact-v1';attention=t.run('attention_full_integer',ax,ah['dims'],tensor=ah['tensor'],input_hash=ah['input_hash']);assert attention.tobytes()==expected_attention.tobytes();control_attention=t.measurements[-1]
   for rows in widths:
    t.wire_codec=DOWN;state=t.run('mlp_prepare_partial_down',x,[n,C,rows],[2.,1e-6],tensor=mh['tensor'],aux=mh['aux'],input_hash=mh['input_hash'])
    finished=t.run('mlp_down_norm_partial_prepared',state,[n,C,rows],[2.,1e-6],tensor=mh['tensor'],aux=mh['aux'],input_hash=mh['input_hash']);assert finished.tobytes()==expected_mlp.tobytes();control_mlp=t.measurements[-1]
    h=dict(mh,version=3,step=t.index,encoding=NAME,op='mlp_finish_attention_full',dims=[n,p,rows]);index=t.index;packet=encode_request(h,state,ax[n*C:])
    try:y=t._run_encoded(h,packet)
    except RuntimeError as error:
     if not is_instruction_limit(error):raise
     records.append(dict(label=label,layer=layer,rows=rows,success=False,error=str(error),failed_request=f'{index:06d}.request.bin'));continue
    expected=np.concatenate([expected_mlp,expected_attention]);assert y.tobytes()==expected.tobytes();call=t.measurements[-1]
    request=d/f'{index:06d}.request.bin';out=d/f'{index:06d}.profile.response.bin';profile=t.command(dict(op='profile',input=str(request),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==expected.tobytes();(d/f'{index:06d}.profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    old=[control_mlp,control_attention];row=dict(label=label,layer=layer,tokens=n,rows=rows,success=True,full_bitwise_equal=True,call=call,control_calls=old,profile=profile,instruction_saving=sum(c['ok']['instructions']for c in old)-call['ok']['instructions'],candid_saving=sum(c['ok']['request_bytes']+c['ok']['reply_bytes']for c in old)-call['ok']['request_bytes']-call['ok']['reply_bytes']);records.append(row);print(json.dumps(row),flush=True)
 verify_module(t,module)
 if sha(ROOT/a.wasm)!=module or hashes!={str(p.relative_to(ROOT)):sha(p)for p in paths}or any(sha(ROOT/p)!=v for p,v in refs.items()):raise ValueError('Proof bookend changed')
 (d/'report.json').write_text(json.dumps(dict(settings=vars(a),module_sha256=module,source_hashes=hashes,reference_hashes=refs,cases=records,measurements=t.measurements,requested_cases=len(labels)*len(layers)*len(widths),completed_cases=sum(c['success']for c in records),failed_cases=sum(not c['success']for c in records),whole_inference_reduction_verified=False),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
