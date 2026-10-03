#!/usr/bin/env python3
"""Exact innovation-log state benchmark, isolated from adopted inference.

Native capture is validation/extraction only. This does not move model arithmetic
onto the inference client. Production capture must be performed inside a query.
"""
import argparse,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import decode,encode

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister');ap.add_argument('--directory',default='artifacts/delta-log/check');ap.add_argument('--helper',default='artifacts/delta-log/native/release/log_args');ap.add_argument('--wasm',default='artifacts/delta-log/wasm/wasm32-unknown-unknown/release/imajev_delta_log_bench.wasm');a=ap.parse_args()
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);helper=ROOT/a.helper;wasm=ROOT/a.wasm;sha=lambda b:hashlib.sha256(b).hexdigest()
 source=ROOT/'artifacts/blake3-v1-prefix';raw=(source/'report.json').read_bytes();r=json.loads(raw)
 paths=list((ROOT/'scripts/delta_log_bench/src').rglob('*.rs'))+[ROOT/'scripts/delta_log_bench/Cargo.toml',ROOT/'scripts/delta_log_bench/Cargo.lock',pathlib.Path(__file__),ROOT/'crates/imajev-runtime/src/lib.rs',ROOT/'crates/imajev-runtime/src/delta_simd.rs',ROOT/'Cargo.lock']
 paths += list((ROOT/'crates/imajev-runtime/src').glob('*.rs'))+[ROOT/'Cargo.toml',ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'client/transport.py',ROOT/'checkpoints/full-int8.manifest.json']
 hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
 def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
 if a.canister:assert status()==sha(wasm.read_bytes())
 cases=[];logs={};calls=0
 for q in r['queries']:
  if q['op']!='delta_stage_bf16':continue
  req=source/'queries'/f'{q["index"]:06d}.request.bin';reply=source/'queries'/f'{q["index"]:06d}.response.bin';h,_=decode(req.read_bytes());layer=int(h['tensor'].split('.layers.')[1].split('.')[0]);sub=d/f'layer-{layer:02d}-{h["dims"][2]:02d}'
  row=json.loads(subprocess.check_output([str(helper),'extract',str(req),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack'),str(reply),str(sub)],text=True,cwd=ROOT));row.update(layer=layer,source_request_sha256=sha(req.read_bytes()),source_reply_sha256=sha(reply.read_bytes()))
  native=json.loads(subprocess.check_output([str(helper),'native',str(sub/'log.bin'),'45','16','true'],text=True,cwd=ROOT));row['native']=native
  log=np.frombuffer((sub/'log.bin').read_bytes(),'<f4').copy();logs[(layer,row['first'])]=log
  if a.canister:
   measurements={}
   for logged in [False,True]:
    ip=sub/('log.bin' if logged else 'capture.bin');arg=sub/f'{logged}.args.bin';subprocess.run([str(helper),'query',str(ip),'45','16',str(logged).lower(),str(arg)],check=True,cwd=ROOT)
    start=time.monotonic();hexreply=subprocess.check_output(['icp','canister','call',a.canister,'restore','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True,cwd=ROOT);elapsed=time.monotonic()-start;calls+=1
    rp=sub/f'{logged}.hex';rp.write_text(hexreply);result=json.loads(subprocess.check_output([str(helper),'decode',str(rp)],text=True,cwd=ROOT));assert result['digest']==native['digest'];result.update(wall_seconds=elapsed,request_candid_bytes=arg.stat().st_size,reply_hex_sha256=sha(hexreply.encode()));measurements['logged' if logged else 'reference']=result
   row['measurements']=measurements
  cases.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('native','measurements')}),flush=True)
 packets=[];full_cases=[]
 for layer in sorted(set(k[0] for k in logs)):
  halves=[logs[(layer,f)] for f in [0,16]];kc=45*8*128;vc=45*16*128
  full=np.concatenate([np.concatenate([v[:kc].reshape(45,8,128) for v in halves],axis=1).ravel(),np.concatenate([v[kc:kc+vc].reshape(45,16,128) for v in halves],axis=1).ravel(),np.concatenate([v[kc+vc:].reshape(45,16) for v in halves],axis=1).ravel()])
  if a.canister:
   sub=d/f'layer-{layer:02d}-full';sub.mkdir(exist_ok=True)
   captured=[np.frombuffer((d/f'layer-{layer:02d}-{f:02d}'/'capture.bin').read_bytes(),'<f4') for f in [0,16]]
   joined=np.concatenate([np.concatenate([v[:kc].reshape(45,8,128) for v in captured],axis=1).ravel(),np.concatenate([v[kc:kc+vc].reshape(45,16,128) for v in captured],axis=1).ravel(),np.concatenate([v[kc+vc:kc+vc+45*16].reshape(45,16) for v in captured],axis=1).ravel(),np.concatenate([v[kc+vc+45*16:].reshape(45,16) for v in captured],axis=1).ravel()])
   (sub/'capture.bin').write_bytes(joined.astype('<f4').tobytes());(sub/'log.bin').write_bytes(full.astype('<f4').tobytes())
   native=json.loads(subprocess.check_output([str(helper),'native',str(sub/'log.bin'),'45','32','true'],text=True,cwd=ROOT));measurements={}
   for logged in [False,True]:
    ip=sub/('log.bin' if logged else 'capture.bin');arg=sub/f'{logged}.args.bin';subprocess.run([str(helper),'query',str(ip),'45','32',str(logged).lower(),str(arg)],check=True,cwd=ROOT)
    start=time.monotonic();hexreply=subprocess.check_output(['icp','canister','call',a.canister,'restore','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True,cwd=ROOT);elapsed=time.monotonic()-start;calls+=1
    rp=sub/f'{logged}.hex';rp.write_text(hexreply);result=json.loads(subprocess.check_output([str(helper),'decode',str(rp)],text=True,cwd=ROOT));assert result['digest']==native['digest'];result.update(wall_seconds=elapsed,request_candid_bytes=arg.stat().st_size,reply_hex_sha256=sha(hexreply.encode()));measurements['logged' if logged else 'reference']=result
   full_cases.append(dict(layer=layer,tokens=45,heads=32,bitwise_equal=True,native=native,measurements=measurements));print(json.dumps(dict(layer=layer,heads=32,bitwise_equal=True)),flush=True)
  for label in ['prefix','617','insufficient','maximum','normal']:
   dr=ROOT/f'artifacts/column32-v1-{label}';rr=json.loads((dr/'report.json').read_text());capture=next(q for q in rr['queries'] if q['op']=='delta_project_capture' and f'.layers.{layer}.' in q['tensor']);header,x=decode((dr/'queries'/f'{capture["index"]:06d}.request.bin').read_bytes());n=header['dims'][0]
   keep=label=='prefix';conv=np.zeros((3,8192),np.float32)
   if label not in ('prefix','normal'):
    with np.load(ROOT/f'artifacts/column32-v1-prefix/queries/states/layer-{layer:02d}.npz') as data:conv=data['conv'].copy()
   proposed=dict(header,op='delta_full_log_integer',dims=[n,32,45 if label not in ('prefix','normal') else 0,int(keep)],aux=[],scalars=[])
   inp=np.concatenate([x[:n*2560],conv.ravel(),full if label not in ('prefix','normal') else np.array([],np.float32)])
   packet=encode(proposed,inp)
   finish=next(q for q in rr['queries'] if q['op']=='delta_project_finish' and f'.layers.{layer}.' in q['tensor']) if label!='normal' else None
   if finish:
    _,out=decode((dr/'queries'/f'{finish["index"]:06d}.response.bin').read_bytes());out=out[:n*2560]
   else:
    oq=next(q for q in rr['queries'] if q['op']=='lora_integer' and q['tensor']==header['tensor'].replace('in_proj_qkv','out_proj'));_,out=decode((dr/'queries'/f'{oq["index"]:06d}.response.bin').read_bytes())
   with np.load(dr/f'queries/states/layer-{layer:02d}.npz') as data:new_conv=data['conv'].copy()
   output=np.concatenate([out,new_conv.ravel(),full if keep else np.array([],np.float32)])
   response=encode(proposed,output)
   packets.append(dict(layer=layer,label=label,tokens=n,input_frame_bytes=len(packet),reply_frame_bytes=len(response),prefix_log_f32_bytes=full.nbytes,input_with_log=label not in ('prefix','normal'),keep_log=keep,fits_frames=max(len(packet),len(response))<2_000_000,logical_input_values=inp.size,logical_reply_values=output.size))
 if a.canister:assert status()==sha(wasm.read_bytes())
 assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
 result=dict(scope='Diagnostic exact prefix innovation log only. Full32 hypothetical packet sizes use real values. Not implemented production fusion, full-model query reduction, or accuracy evidence.',model=r['model'],pack_hash=r['pack_hash'],canister=a.canister,wasm_sha256=sha(wasm.read_bytes()) if a.canister else None,helper_sha256=sha(helper.read_bytes()),source_report_sha256=sha(raw),source_hashes=hashes,ordinary_queries=calls,module_status_update_reads=2 if a.canister else 0,cases=cases,full_cases=full_cases,packets=packets,all_packets_fit=all(p['fits_frames'] for p in packets))
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),ordinary_queries=calls,packets=len(packets),all_packets_fit=result['all_packets_fit'])),flush=True)
if __name__=='__main__':main()
