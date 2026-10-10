#!/usr/bin/env python3
"""Same-module ordinary-query control and client-held terminal replay.

Only frames saved canister inputs/outputs; no host numerical inference.
"""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,encode,atomic
from full_inference import JournalTransport
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['canister','wasm','source','directory']:ap.add_argument('--'+name,required=True)
 a=ap.parse_args();source=ROOT/a.source;dest=ROOT/a.directory;dest.mkdir(parents=True,exist_ok=True)
 sha=lambda b:hashlib.sha256(b).hexdigest()
 report=json.loads((source/'report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
 assert report['model']==m['model'] and report['pack_hash']==m['pack_hash']
 module=sha((ROOT/a.wasm).read_bytes());assert report['wasm_sha256']==module
 q=next(q for q in report['queries'] if q['op']=='terminal_attention_mlp_integer')
 assert 'decision' in q['ok'] and report['decision_query']['included_in_terminal_query']
 options=q['decision_options'];request=source/'queries'/f"{q['index']:06d}.request.bin";expected=source/'queries'/f"{q['index']:06d}.response.bin"
 r,x=decode(request.read_bytes());decision=q['ok']['decision']
 files=[pathlib.Path(__file__),ROOT/'client/transport.py',ROOT/'client/full_inference.py',ROOT/'crates/imajev-client/src/main.rs',ROOT/'canisters/inference/src/lib.rs',ROOT/'Cargo.lock']
 hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in files};binary=sha((ROOT/'target/release/imajev-client').read_bytes())
 keys=['value','abstained','raw_logits','probabilities','unknown_probability','calibration_version']
 rows=[];t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash'])
 try:
  verify_module(t,module)
  output=dest/'separate-terminal.response.bin';step=t.command(dict(op='step',input=str(request),output=str(output)))
  assert output.read_bytes()==expected.read_bytes();rows.append(dict(method='step',**step))
  output=dest/'fused-terminal.response.bin';fused=t.command(dict(op='terminal_step_decision',input=str(request),output=str(output),options=options));rows.append(dict(method='terminal_step_decision',**fused))
  assert output.read_bytes()==expected.read_bytes();assert all(fused['ok']['decision'][key]==decision[key]for key in keys)
  errors=[]
  bad=dict(r,tensor='model.language_model.layers.30.self_attn.q_proj.weight');badfile=dest/'wrong-stage.request.bin';atomic(badfile,encode(bad,x))
  for label,path,choices in [('duplicate-options',request,[options[0],options[0]]),('wrong-stage',badfile,options)]:
   try:t.command(dict(op='terminal_step_decision',input=str(path),output=str(dest/(label+'.bin')),options=choices))
   except RuntimeError as error:errors.append(dict(case=label,error=str(error)))
   else:raise AssertionError('invalid terminal decision accepted')
  verify_module(t,module)
 finally:t.close()
 # Replay actual canister checkpoint without any query or client-side inference.
 replaydir=dest/'replay';replaydir.mkdir(exist_ok=True)
 for suffix in ['request.bin','response.bin','metric.json']:
  name=f"{q['index']:06d}.{suffix}";(replaydir/name).write_bytes((source/'queries'/name).read_bytes())
 def replay(choices):
  t=JournalTransport.__new__(JournalTransport);t.directory=replaydir;t.index=q['index'];t.measurements=[];t.replayed=0;t.model=m['model'];t.pack_hash=m['pack_hash'];t.input_hash=r['input_hash'];t.frame_version=r['version'];t.wire_codec=r['encoding'];t.fuse_terminal_decision=True;t.decision_options=choices
  def forbidden(*_):raise AssertionError('checkpoint replay called the replica')
  t.command=forbidden
  result=t.run(r['op'],x,r['dims'],r['scalars'],tensor=r['tensor'],aux=r.get('aux',[]))
  return t,result
 replayed,values=replay(options);assert values.tobytes()==decode(expected.read_bytes())[1].tobytes();assert all(replayed.terminal_decision[k]==decision[k]for k in keys);assert replayed.replayed==1
 try:replay(list(reversed(options)))
 except ValueError:pass
 else:raise AssertionError('changed option order accepted on replay')
 assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in files};assert binary==sha((ROOT/'target/release/imajev-client').read_bytes())
 data=dict(scope=__doc__,wasm_sha256=module,canister=a.canister,source_report_sha256=sha((source/'report.json').read_bytes()),request_sha256=sha(request.read_bytes()),client_binary_sha256=binary,source_hashes=hashes,ordinary_queries=4,successful_control_queries=2,rejected_invalid_queries=errors,cases=rows,state_byte_equal=True,decision_equal=True,replay_without_query=True,changed_options_replay_rejected=True)
 (dest/'report.json').write_text(json.dumps(data,indent=2)+'\n')
 with zipfile.ZipFile(dest/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(state_byte_equal=True,decision_equal=True,replay_without_query=True,ordinary_queries=4)))
if __name__=='__main__':main()
