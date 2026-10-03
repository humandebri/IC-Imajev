#!/usr/bin/env python3
"""Compare full32-head Delta to the old two-query path, separately per backend."""
import argparse,hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import decode,encode,Transport
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister');ap.add_argument('--wasm',default='artifacts/delta-full-log/full.wasm');ap.add_argument('--directory',default='artifacts/delta-full-log/partial');ap.add_argument('--layers',default='0,22,30');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());helper=ROOT/'target/release/primitive';verifier=ROOT/'target/release/delta_log_verify';sha=lambda b:hashlib.sha256(b).hexdigest();wasm=ROOT/a.wasm
 t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d/'queries',m['pack_hash']) if a.canister else None
 def native(req,out):subprocess.run([str(helper),str(req),str(out),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True,cwd=ROOT);return decode(out.read_bytes())[1]
 hashes=json.loads((ROOT/'artifacts/delta-full-log/source-hashes.json').read_text());hashes[str(pathlib.Path(__file__).relative_to(ROOT))]=sha(pathlib.Path(__file__).read_bytes())
 assert all(sha((ROOT/k).read_bytes())==v for k,v in hashes.items())
 cases=[]
 try:
  if t:verify_module(t,sha(wasm.read_bytes()))
  for label in ['prefix','617','insufficient','maximum']:
   source=ROOT/f'artifacts/column32-v1-{label}';report_raw=(source/'report.json').read_bytes();r=json.loads(report_raw)
   for layer in map(int,a.layers.split(',')):
    qs=[q for q in r['queries'] if f'.layers.{layer}.' in q['tensor']];first=next(q for q in qs if q['op']=='delta_project_capture');last=next(q for q in qs if q['op']=='delta_project_finish');fp=source/'queries'/f'{first["index"]:06d}.request.bin';lp=source/'queries'/f'{last["index"]:06d}.request.bin';h,x=decode(fp.read_bytes());lh,lx=decode(lp.read_bytes());n=h['dims'][0];keep=label=='prefix';p=0 if keep else 45
    state_path=ROOT/f'artifacts/column32-v1-prefix/queries/states/layer-{layer:02d}.npz'
    with np.load(state_path,allow_pickle=False) as f:conv=np.zeros((3,8192),np.float32) if keep else f['conv'].copy();initial=np.zeros((32,128,128),np.float32) if keep else f['delta'].copy()
    logs=[np.frombuffer((ROOT/f'artifacts/delta-log/check/layer-{layer:02d}-{i:02d}/log.bin').read_bytes(),'<f4') for i in [0,16]];kc=45*8*128;vc=45*16*128
    log=np.concatenate([np.concatenate([v[:kc].reshape(45,8,128) for v in logs],axis=1).ravel(),np.concatenate([v[kc:kc+vc].reshape(45,16,128) for v in logs],axis=1).ravel(),np.concatenate([v[kc+vc:].reshape(45,16) for v in logs],axis=1).ravel()]) if p else np.array([],np.float32)
    header=dict(h,op='delta_full_log_integer',dims=[n,32,p,int(keep)],aux=[],scalars=[]);inp=np.concatenate([x[:n*2560],conv.ravel(),log]);sub=d/f'{label}-{layer:02d}';sub.mkdir(exist_ok=True);req=sub/'full.request.bin';req.write_bytes(encode(header,inp))
    row=dict(label=label,layer=layer,tokens=n,source_report_sha256=sha(report_raw),input_sha256=sha(req.read_bytes()))
    if t:
     out=sub/'full.response.bin';result=t.command(dict(op='step',input=str(req),output=str(out)));got=decode(out.read_bytes())[1]
     old=decode((source/'queries'/f'{last["index"]:06d}.response.bin').read_bytes())[1][:n*2560]
     with np.load(source/f'queries/states/layer-{layer:02d}.npz') as f:expected_conv=f['conv'].copy();expected_state=f['delta'].copy() if keep else None
     assert np.array_equal(got[:n*2560].view(np.uint32),old.view(np.uint32));assert np.array_equal(got[n*2560:n*2560+24576].view(np.uint32),expected_conv.ravel().view(np.uint32));row.update(wasm_bitwise_equal=True,query=result)
     if keep:
      logp=sub/'saved-log.bin';logp.write_bytes(got[n*2560+24576:].astype('<f4').tobytes());sp=sub/'restored-state.bin';subprocess.run([str(verifier),str(n),str(logp),str(sp)],check=True,cwd=ROOT);assert sp.read_bytes()==expected_state.astype('<f4').tobytes();row['wasm_log_state_bitwise_equal']=True
    else:
     got=native(req,sub/'full.native.bin');old_first=native(fp,sub/'old-first.native.bin');tail=n*2048+3*16*256+16*16384*int(keep);next_input=lx.copy();next_input[:n*2762]=old_first[tail:];next_input[-n*2048:]=old_first[:n*2048];second=sub/'old-second.request.bin';second.write_bytes(encode(lh,next_input));old_last=native(second,sub/'old-second.native.bin')
     assert np.array_equal(got[:n*2560].view(np.uint32),old_last[:n*2560].view(np.uint32));expected_conv=np.empty((3,8192),np.float32)
     for start,output in [(0,old_first),(16,old_last)]:
      indices=np.concatenate([np.arange(start//2*128,(start//2+8)*128),np.arange(2048+start//2*128,2048+(start//2+8)*128),np.arange(4096+start*128,4096+(start+16)*128)]);offset=n*2560 if start else n*2048;expected_conv[:,indices]=output[offset:offset+3*4096].reshape(3,4096)
     assert np.array_equal(got[n*2560:n*2560+24576].view(np.uint32),expected_conv.ravel().view(np.uint32));row['native_bitwise_equal']=True
     if keep:
      expected=np.concatenate([old_first[n*2048+12288:tail],old_last[n*2560+12288:]]);logp=sub/'saved-log.bin';logp.write_bytes(got[n*2560+24576:].astype('<f4').tobytes());sp=sub/'restored-state.bin';subprocess.run([str(verifier),str(n),str(logp),str(sp)],check=True,cwd=ROOT);assert sp.read_bytes()==expected.astype('<f4').tobytes();row['native_log_state_bitwise_equal']=True
    cases.append(row);print(json.dumps(row),flush=True)
  if t:verify_module(t,sha(wasm.read_bytes()))
  assert all(sha((ROOT/k).read_bytes())==v for k,v in hashes.items())
  (d/'report.json').write_text(json.dumps(dict(cases=cases,wasm_sha256=sha(wasm.read_bytes()) if t else None,model=m['model'],pack_hash=m['pack_hash'],scope='Full Delta boundary comparison only. Native compared to old native and Wasm compared to old Wasm separately; full graph proof required.',ordinary_queries=len(cases) if t else 0,certified_module_reads=2 if t else 0,source_hashes=hashes,helper_sha256=sha(helper.read_bytes()),verifier_sha256=sha(verifier.read_bytes())),indent=2)+'\n')
 finally:
  if t:t.close()
if __name__=='__main__':main()
