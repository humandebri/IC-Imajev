#!/usr/bin/env python3
"""Measure two separate queries and fused tail on one module; no host inference."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--source',default='artifacts/prefix_codec/full-address-reuse-proof');a=ap.parse_args();dest=ROOT/a.directory;dest.mkdir(parents=True,exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());module=sha((ROOT/a.wasm).read_bytes())
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__),ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'canisters/inference/Cargo.toml',ROOT/'Cargo.lock',ROOT/'crates/imajev-client/src/main.rs'];hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
keys=['value','abstained','raw_logits','probabilities','unknown_probability','calibration_version'];cases=[]
for label in ['617','insufficient','maximum']:
 source=ROOT/a.source/label;raw=(source/'report.json').read_bytes();r=json.loads(raw);assert r['model']==m['model'] and r['pack_hash']==m['pack_hash'];mq,tq=r['queries'][-2:];assert mq['op']=='mlp_full_integer' and tq['op']=='terminal_attention_mlp_integer'
 def frame(q,suffix):return source/'queries'/f"{q['index']:06d}.{suffix}.bin"
 mh,x=decode(frame(mq,'request').read_bytes());th,tx=decode(frame(tq,'request').read_bytes());_,my=decode(frame(mq,'response').read_bytes());_,ty=decode(frame(tq,'response').read_bytes());n,p=th['dims'];count=n*2560;assert x.size==2*count and tx.size==count+2560+p*2048;assert tx[:count].tobytes()==my[count:].tobytes() and tx[count:count+2560].tobytes()==my[count-2560:count].tobytes()
 data=np.concatenate([x,tx[count+2560:]]);options=tq['decision_options'];t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),dest/label,m['pack_hash'],wire_codec='bf16-block256-exact-v1',frame_version=2);t.frame_checksum='blake3';t.fuse_terminal_decision=True;t.decision_options=options
 try:
  verify_module(t,module)
  first=t.run('mlp_full_integer',x,[n,2560],[2.,1e-6],tensor=mh['tensor'],input_hash=mh['input_hash'],aux=mh['aux']);assert first.tobytes()==my.tobytes()
  last=t.run('terminal_attention_mlp_integer',tx,[n,p],tensor=th['tensor'],input_hash=th['input_hash']);assert last.tobytes()==ty.tobytes();assert all(t.terminal_decision[k]==tq['ok']['decision'][k]for k in keys)
  row=dict(label=label,suffix_tokens=n,prefix_tokens=p,source_report_sha256=sha(raw),control_queries=list(t.measurements),control_instructions=sum(q['ok']['instructions']for q in t.measurements),control_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes']for q in t.measurements))
  try:
   both=t.run('terminal_tail_integer',data,[n,p],tensor=mh['tensor'],input_hash=mh['input_hash']);assert both.size==count+ty.size and both[:count].tobytes()==my[:count].tobytes() and both[count:].tobytes()==ty.tobytes();assert all(t.terminal_decision[k]==tq['ok']['decision'][k]for k in keys)
   fused=t.measurements[-1];row.update(full_bitwise_equal=True,fused=fused,instruction_saving=row['control_instructions']-fused['ok']['instructions'],byte_saving=row['control_bytes']-fused['ok']['request_bytes']-fused['ok']['reply_bytes'])
  except RuntimeError as e:
   if not is_instruction_limit(e):raise
   row.update(full_bitwise_equal=False,fused_failure=str(e));(dest/label/'fused-failure.txt').write_text(str(e)+'\n')
  verify_module(t,module);cases.append(row);print(json.dumps(row),flush=True)
 finally:t.close()
assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in paths}
with zipfile.ZipFile(dest/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in paths:z.write(p,str(p.relative_to(ROOT)))
(dest/'report.json').write_text(json.dumps(dict(scope=__doc__,wasm_sha256=module,source_hashes=hashes,cases=cases),indent=2)+'\n')
