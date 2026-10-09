#!/usr/bin/env python3
"""Run the server-held update graph from token IDs, verify each layer and decision."""
import argparse,hashlib,json,pathlib,subprocess,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bits(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--prepare-prefix',action='store_true');ap.add_argument('--cases',default='617,insufficient,maximum');a=ap.parse_args()
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=False);m=sha(ROOT/a.wasm);base=ROOT/'artifacts/single-quad/full-proof-v2/prefix/queries';packets=ROOT/'artifacts/single-quad/full-proof-v2/packets';metadata=json.loads((base/'cache.json').read_text());fixture=json.loads((ROOT/'artifacts/reference-serving.json').read_text());report=dict(module_sha256=m,canister='6eydd-o3777-77775-aaama-cai',prefix_preparation=[],cases=[],reference_hashes={})
 bridge=ROOT/'artifacts/update-inference/client-target/release/imajev-client';report['bridge_sha256']=sha(bridge);report['script_sha256']=sha(pathlib.Path(__file__))
 p=subprocess.Popen([str(bridge),'http://localhost:8001/',report['canister'],str(ROOT/'artifacts/imajev-local.pem')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
 def call(cmd):
  p.stdin.write(json.dumps(cmd)+'\n');p.stdin.flush();line=p.stdout.readline()
  if not line:raise RuntimeError('bridge exited')
  r=json.loads(line)
  if 'error'in r:raise RuntimeError(r['error'])
  return r
 def ref(path):report['reference_hashes'][str(path.relative_to(ROOT))]=sha(path);return path
 try:
  assert call(dict(op='module_hash'))['ok']['module_hash']==m
  if a.prepare_prefix:
   pd=d/'prefix';pd.mkdir()
   for i in range(32):
    source=ref(base/'states'/f'layer-{i:02d}.npz')
    with np.load(source,allow_pickle=False)as z:
     if i%4==3:v=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()]);packet=None
     else:v=z['conv'].ravel();packet=ref(packets/f'layer-{i:02d}.npf1')
    path=pd/f'{i:02d}.f32';np.asarray(v,dtype='<f4').tofile(path);cmd=dict(op='update_prefix',layer=i,values=str(path))
    if packet:cmd['packet']=str(packet)
    r=call(cmd);report['prefix_preparation'].append(dict(layer=i,**r));print(json.dumps(dict(prefix_layer=i,seconds=r['wall_seconds'])),flush=True)
  for name in a.cases.split(','):
   index={'617':0,'insufficient':19,'maximum':11}[name];record=fixture['records'][index];ids=record['token_ids'];assert ids[:45]==metadata['token_ids'];directory=d/name;directory.mkdir();old=ROOT/f'artifacts/output-pairs-v2-{name}';reference=json.loads(ref(old/'report.json').read_text());options=record['options'];rows=[];before=call(dict(op='balance_status'));started=time.perf_counter();cmd=dict(diagnostics=True,op='update_infer_start',ids=ids[45:],options=options)
   while True:
    r=call(cmd);progress=r['ok']['progress'];rows.append(r);(directory/f'{len(rows):02d}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(case=name,call=len(rows),stage=progress['stage'],instructions=progress['instructions'],seconds=r['wall_seconds'])),flush=True)
    if progress['done']:break
    cmd=dict(diagnostics=True,op='update_infer_continue',id=progress['id'],stage=progress['stage'])
   elapsed=time.perf_counter()-started;after=call(dict(op='balance_status'))
   for i in range(32):
    path=ref(old/'queries'/f'layer-{i:02d}.npy');v=np.load(path,allow_pickle=False)
    # The legacy reference preserves prefix rows at ordinary layers; terminal
    # readout exports only the final token of layer31.
    if i<31:v=v[45:]
    assert bits(v)==progress['hidden_hashes'][i],('hidden',name,i,v.shape)
    with np.load(ref(old/'queries/states'/f'layer-{i:02d}.npz'),allow_pickle=False)as z:
     if i%4==3:v=np.concatenate([z['keys'][45:].ravel(),z['values'][45:].ravel()])
     else:v=z['conv'].ravel()
    assert bits(v)==progress['state_hashes'][i],('state',name,i)
   expected=reference['decision_query']['ok']['decision'];actual=progress['decision'];assert {k:v for k,v in expected.items()if k!='instructions'}=={k:v for k,v in actual.items()if k!='instructions'},'decision'
   assert bits(np.asarray(progress['final_hidden'],dtype=np.float32))==bits(np.load(ref(old/'final-hidden.npy'),allow_pickle=False)),'final norm'
   row=dict(case=name,tokens=len(ids),suffix_tokens=len(ids)-45,update_calls=len(rows),total_handler_instructions=sum(r['ok']['progress']['instructions']for r in rows),max_handler_instructions=max(r['ok']['progress']['instructions']for r in rows),total_candid_bytes=sum(r['ok']['request_bytes']+r['ok']['reply_bytes']for r in rows),call_wall_seconds=sum(r['wall_seconds']for r in rows),loop_seconds=elapsed,observed_balance_decrease=int(before['ok']['cycles'])-int(after['ok']['cycles']),all_32_hidden_and_state_hashes_equal=True,decision_equal=True,final_norm_equal=True,balance_before=before,balance_after=after,max_observed_heap_bytes=max(r['ok']['progress']['heap_pages']*65536 for r in rows))
   report['cases'].append(row);(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
  # Rejected continuation/start must leave the completed session intact.
  boundary=[]
  for cmd in [dict(diagnostics=True,op='update_infer_continue',id=progress['id'],stage=progress['stage']),dict(diagnostics=True,op='update_infer_start',ids=[],options=options)]:
   try:call(cmd)
   except RuntimeError as error:
    assert str(error) in ('session progress mismatch','suffix token bounds'),str(error)
    boundary.append(dict(command=cmd['op'],rejected=str(error)))
   else:raise AssertionError('invalid update accepted')
  report['boundary_verification_updates']=boundary
  assert call(dict(op='module_hash'))['ok']['module_hash']==m
  assert all(sha(ROOT/p)==h for p,h in report['reference_hashes'].items())
  (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 except Exception as e:
  (d/'failure.json').write_text(json.dumps(dict(error=str(e),partial=report),indent=2)+'\n');raise
 finally:p.stdin.close();p.wait(timeout=15)
if __name__=='__main__':main()
