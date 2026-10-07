#!/usr/bin/env python3
"""Snapshot protected dense-state diagnostic on all three unchanged inputs."""
from pathlib import Path
import hashlib,json,struct,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module
from check_delta_capture import check
TARGET='6eydd-o3777-77775-aaama-cai'
BASELINE='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
B=ROOT/'artifacts/delta-capture-v2/build';D=B.parent/'proof'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def bits(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False)
 build=json.loads((B/'report.json').read_text());candidate=sha(B/'full.wasm');assert candidate==build['wasm_sha256']
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in build[key].items())
 tools=json.loads((ROOT/'artifacts/paid-update-v1/tools/report.json').read_text());cmd=tools['args_command'][:]
 cmd[cmd.index('--edition=2021')+1]=str(ROOT/'scripts/delta_capture_args.rs');cmd[cmd.index('-o')+1]=str(D/'decode')
 subprocess.run(cmd,check=True,cwd=ROOT)
 did=D/'capture.did';did.write_text('service:{delta_capture_chunk:(nat32,nat32)->(variant{Ok:vec nat8;Err:text}) query;reset_update_prefix:()->(variant{Ok;Err:text});}')
 sources=[Path(__file__),ROOT/'scripts/check_delta_capture.py',ROOT/'scripts/delta_capture_args.rs',ROOT/'scripts/delta_capture.rs',D/'decode',did,B/'report.json',B/'full.wasm',ROOT/'artifacts/text-short-v2/inputs.json',ROOT/'checkpoints/full-int8.manifest.json']
 hashes={str(p.relative_to(ROOT)):sha(p)for p in sources};write(D/'sources.json',hashes)
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());events=[];captures=[];cases=[]
 def icp(*args):
  start=time.monotonic();raw=subprocess.check_output(['icp','canister',*args,'--network','local','--identity','imajev-local'],text=True,cwd=ROOT)
  events.append(dict(args=list(args),seconds=time.monotonic()-start,output=raw));write(D/'operations.json',events);return raw.strip()
 def bridge(path):return Transport(m['model'],'http://localhost:8001/',TARGET,str(ROOT/'artifacts/imajev-local.pem'),path,m['pack_hash'],bridge_binary=str(ROOT/'artifacts/query-packing-v3/build/imajev-client'))
 t=bridge(D/'before')
 try:
  verify_module(t,BASELINE);cache=t.command(dict(op='weight_cache_status'))['ok']['cache'];pack=t.command(dict(op='pack_status'))['ok']
  assert cache['bytes']==4065416192 and len(cache['names'])==721 and pack['ready']
 finally:t.close()
 write(D/'before.json',dict(module=BASELINE,cache=cache,pack=pack))
 snapshot=None;stopped=False;restored=False
 def fetch(directory,offset,length,label):
  p=directory/(label+'.reply.hex');out=directory/(label+'.bin')
  raw=icp('call',TARGET,'delta_capture_chunk',f'({offset}:nat32,{length}:nat32)','--candid',str(did),'--query','--output','hex');p.write_text(raw+'\n')
  # Keep large replies in saved files, rather than the operations ledger.
  events[-1]['output']=dict(saved=str(p.relative_to(ROOT)),sha256=sha(p));write(D/'operations.json',events)
  subprocess.run([str(D/'decode'),str(p),str(out)],check=True);assert out.stat().st_size==length
  return out.read_bytes()
 try:
  icp('stop',TARGET);stopped=True;snapshot=icp('snapshot','create',TARGET,'--quiet');write(D/'snapshot.json',dict(id=snapshot))
  print('snapshot saved',flush=True)
  icp('install',TARGET,'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes');icp('start',TARGET);stopped=False
  with (D/'preparation.log').open('w')as log:subprocess.run([sys.executable,str(ROOT/'scripts/prepare_weight_cache.py'),'--canister',TARGET,'--wasm',str(B/'full.wasm'),'--directory',str(D/'preparation'),'--include-f32','--require-prepared-rope','--require-prepared-activation','--require-all-output-pairs'],check=True,cwd=ROOT,stdout=log,stderr=log)
  print('weights ready',flush=True)
  with (D/'fixed-prefix.log').open('w')as log:subprocess.run([sys.executable,str(ROOT/'scripts/prepare_fixed_prefix_states.py'),'--canister',TARGET,'--wasm',str(B/'full.wasm'),'--directory',str(D/'fixed-prefix')],check=True,cwd=ROOT,stdout=log,stderr=log)
  fixture=json.loads((ROOT/'artifacts/text-short-v2/inputs.json').read_text())['records']
  for bank,names in [('voting',['617','620']),('common',['653'])]:
   if bank=='common':assert 'Ok'in icp('call',TARGET,'reset_update_prefix','()','--candid',str(did),'--output','candid')
   directory=D/bank;directory.mkdir()
   base=ROOT/('artifacts/voting-template-prefix-v1/prefix/queries'if bank=='voting'else'artifacts/query-packing-v3/prefix-v2/queries')
   packets=ROOT/('artifacts/voting-template-prefix-v1/packets'if bank=='voting'else'artifacts/query-packing-v3/packets-v2')
   metadata=json.loads((base/'cache.json').read_text());prefix=len(metadata['token_ids']);t=bridge(directory/'transport')
   try:
    verify_module(t,candidate)
    for layer in range(32):
     with np.load(base/'states'/f'layer-{layer:02d}.npz',allow_pickle=False)as z:v=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()])if layer%4==3 else z['conv'].ravel()
     p=directory/f'prefix-{layer:02d}.f32';np.asarray(v,dtype='<f4').tofile(p);cmd=dict(op='update_prefix',layer=layer,values=str(p))
     if layer%4!=3:cmd['packet']=str(packets/f'layer-{layer:02d}.npf1')
     t.command(cmd)
    for name in names:
     record=fixture[{'617':0,'620':1,'653':2}[name]];assert record['token_ids'][:prefix]==metadata['token_ids']
     case=directory/name;case.mkdir();cmd=dict(op='update_infer_start',ids=record['token_ids'][prefix:],options=record['options']);previous=0
     while True:
      reply=t.command(cmd);progress=reply['ok']['progress'];stage=progress['stage'];assert stage==previous+1
      write(case/f'stage-{stage:02d}.json',reply);previous=stage
      if stage%2==1 and (stage//2)%4!=3:
       layer=stage//2;cd=case/f'layer-{layer:02d}';cd.mkdir();header=fetch(cd,0,24,'header');assert header[:4]==b'DLC1'
       n,layout,heads,step=struct.unpack_from('<IIIQ',header,4);assert n==len(record['token_ids'])-prefix and layout==1 and heads==32 and step==layer*2
       length=24+4*(2*32*16384+4*32*n*128+2*32*n)
       raw=b''.join(fetch(cd,o,min(524288,length-o),f'chunk-{o:08d}')for o in range(0,length,524288));assert raw[:24]==header
       p=cd/'capture.bin';p.write_bytes(raw);result=check(p);result.update(case=name,reply_hashes={str(q.relative_to(ROOT)):sha(q)for q in cd.glob('*.reply.hex')})
       captures.append(result);write(cd/'verified.json',result);write(D/'captures.json',captures)
       print(json.dumps(dict(case=name,layer=layer,dense_state_bit_equal=True,output_bit_equal=True)),flush=True)
      if progress['done']:break
      cmd=dict(op='update_infer_continue',id=progress['id'],stage=stage)
     old=ROOT/f'artifacts/boomdao-current-v1/{name}-r1'
     for i in range(32):
      p=old/'queries'/f'layer-{i:02d}.npy'
      if p.exists():
       v=np.load(p,allow_pickle=False);assert bits(v[prefix:]if i<31 else v)==progress['hidden_hashes'][i],('hidden',name,i)
      else:assert i==30
      with np.load(old/'queries/states'/f'layer-{i:02d}.npz',allow_pickle=False)as z:v=np.concatenate([z['keys'][prefix:].ravel(),z['values'][prefix:].ravel()])if i%4==3 else z['conv'].ravel()
      assert bits(v)==progress['state_hashes'][i],('state',name,i)
     expected=json.loads((old/'report.json').read_text())['decision_query']['ok']['decision'];actual=progress['decision']
     assert {k:v for k,v in expected.items()if k!='instructions'}=={k:v for k,v in actual.items()if k!='instructions'}
     assert bits(progress['final_hidden'])==bits(np.load(old/'final-hidden.npy',allow_pickle=False))
     cases.append(dict(case=name,dense_layers_checked=24,hidden_layers_checked=[i for i in range(32)if i!=30],state_hashes_checked=32,decision_equal=True,final_norm_equal=True));write(D/'cases.json',cases)
   finally:t.close()
  assert len(captures)==72 and len(cases)==3
  assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in sources}
  write(D/'report.json',dict(complete=False,module=candidate,cases=cases,captures=captures,source_hashes=hashes,scope='Correctness diagnostic only. Ordered independent F32 Delta recurrence verifies final dense state and pre-BF output; layer30 lacks historical hidden reference. No performance claim.'))
 finally:
  if snapshot:
   if not stopped:icp('stop',TARGET);stopped=True
   icp('snapshot','restore',TARGET,snapshot);icp('start',TARGET);stopped=False;t=bridge(D/'restored')
   try:
    verify_module(t,BASELINE);assert t.command(dict(op='weight_cache_status'))['ok']['cache']==cache;assert t.command(dict(op='pack_status'))['ok']==pack
   finally:t.close()
   icp('snapshot','delete',TARGET,snapshot);restored=True;write(D/'restored.json',dict(module=BASELINE,cache_equal=True,pack_equal=True,snapshot_deleted=True));print('baseline restored',flush=True)
  elif stopped:icp('start',TARGET)
 assert restored;r=json.loads((D/'report.json').read_text());r.update(complete=True,baseline_restored=True);write(D/'report.json',r)
if __name__=='__main__':main()
