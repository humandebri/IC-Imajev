#!/usr/bin/env python3
"""Run from token IDs to dedicated decision using only ordinary canister queries."""
import argparse,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import JournalTransport,TextGraph
from prefix_inference import PrefixTextGraph,load_cache,file_hash,graph_hash,verify_module
from transport import encode,atomic,Transport

def main():
 overall_start=time.perf_counter()
 ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://localhost:8001/');ap.add_argument('--canister',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--prepare-prefix',action='store_true');ap.add_argument('--prefix-tokens',type=int,default=45);ap.add_argument('--manifest',default='checkpoints/full-int8.manifest.json');ap.add_argument('--reference',default='artifacts/reference-first.json');ap.add_argument('--record',type=int,default=0);ap.add_argument('--directory',default='artifacts/full-int8-efficient-run');ap.add_argument('--layers',type=int,default=32);ap.add_argument('--wire-codec',choices=['f32','bf16-exact','int8-block256-v1'],default='bf16-exact');ap.add_argument('--arithmetic',choices=['f32','int8'],default='f32');ap.add_argument('--compact-lossless',action='store_true');ap.add_argument('--fuse-add-norm',action='store_true');ap.add_argument('--fuse-mlp',action='store_true');ap.add_argument('--group-projections',action='store_true');ap.add_argument('--projection-group-cap',type=int,choices=[1,2,3],default=2);ap.add_argument('--attention-head-cap',type=int,choices=[1,2,4,8],default=1);ap.add_argument('--delta-head-cap',type=int,choices=[1,2,4,8,16],default=1);ap.add_argument('--row-cap',type=int,default=1536);ap.add_argument('--work-cap',type=int,default=450000000);ap.add_argument('--token-cap',type=int);args=ap.parse_args()
 if args.token_cap is None:args.token_cap=512 if args.arithmetic=='int8' else 132
 m=json.loads((ROOT/args.manifest).read_text());fixture=json.loads((ROOT/args.reference).read_text());record=fixture['records'][args.record];token_ids=record['token_ids'];input_hash=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
 if not 1<=args.layers<=32:raise SystemExit('layers must be 1..32')
 if fixture['model_lock_sha256']!=m['model']:raise SystemExit('reference/model lock mismatch')
 if input_hash!=record['input_sha256']:raise SystemExit('token identity mismatch')
 if args.layers!=32 or args.arithmetic!='int8' or args.wire_codec!='bf16-exact':raise SystemExit('prefix path requires full integer/lossless graph')
 if args.prepare_prefix:
  if not 1<=args.prefix_tokens<len(token_ids):raise SystemExit('prefix token count')
  token_ids=token_ids[:args.prefix_tokens];input_hash=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
 wasm_hash=file_hash(ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm')
 verifier=Transport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/module-verification',m['pack_hash'])
 try:verify_module(verifier,wasm_hash)
 finally:verifier.close()
 cache_start=time.perf_counter()
 cache=None if args.prepare_prefix else load_cache(ROOT/args.cache,m,wasm_hash)
 cache_load_seconds=time.perf_counter()-cache_start
 directory=ROOT/args.directory;directory.mkdir(parents=True,exist_ok=True)
 session=dict(version=1,graph='qwen35-text-bf16-v1',model=m['model'],pack_hash=m['pack_hash'],input_hash=input_hash,url=args.url,canister=args.canister,row_cap=args.row_cap,token_cap=args.token_cap,work_cap=args.work_cap,wire_codec=args.wire_codec)
 if args.arithmetic=='int8':
  if args.wire_codec!='bf16-exact' or args.group_projections:raise SystemExit('integer evaluation requires bf16-exact, without group-projections')
  session["arithmetic"]="int8-block256-base-f32-lora-v1"
  session["integer_scheduler"]="aligned-token8-v1"
 if args.fuse_mlp:
  if args.arithmetic!='int8' or args.wire_codec!='bf16-exact':raise SystemExit('fuse-mlp requires int8 arithmetic and lossless wire')
  session['fuse_mlp']='gate-up-swiglu-v1'
  session['mlp_scheduler']='work-budget-v2'
 if args.fuse_add_norm:
  if args.wire_codec!='bf16-exact':raise SystemExit('fuse-add-norm requires lossless wire')
  session['fuse_add_norm']=True
 if args.compact_lossless:
  if args.wire_codec!='bf16-exact':raise SystemExit('compact-lossless requires bf16-exact wire')
  session["compact_lossless"]=True
 if args.group_projections:session["group_projections"]=True
 if args.projection_group_cap!=2:
  if not args.group_projections:raise SystemExit('projection-group-cap requires group-projections')
  session["projection_group_cap"]=args.projection_group_cap
 if args.delta_head_cap>1:session["delta_head_cap"]=args.delta_head_cap
 if args.attention_head_cap>1:session["attention_head_cap"]=args.attention_head_cap
 session['prefix_mode']='prepare' if args.prepare_prefix else 'continue'
 session['prefix_identity']=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest() if args.prepare_prefix else cache['identity']
 session['prefix_graph_sha256']=graph_hash()
 session['wasm_sha256']=wasm_hash
 if args.prepare_prefix and (ROOT/args.cache).resolve()!=(directory/'queries').resolve():raise SystemExit('prepare cache must equal directory/queries')
 path=directory/'session.json'
 if path.exists() and json.loads(path.read_text())!=session:raise SystemExit('Session mismatch; use a fresh directory')
 atomic(path,(json.dumps(session,indent=2)+'\n').encode())
 t=JournalTransport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),directory/'queries',m['pack_hash'],input_hash=input_hash,wire_codec='' if args.wire_codec=='f32' else args.wire_codec)
 t.group_projections=args.group_projections
 t.projection_group_cap=args.projection_group_cap
 graph=(TextGraph if args.prepare_prefix else PrefixTextGraph)(t,m,**({} if args.prepare_prefix else dict(cache=cache)),row_cap=args.row_cap,token_cap=args.token_cap,work_cap=args.work_cap,delta_head_cap=args.delta_head_cap,attention_head_cap=args.attention_head_cap,compact_lossless=args.compact_lossless,arithmetic=args.arithmetic,fuse_add_norm=args.fuse_add_norm,fuse_mlp=args.fuse_mlp);start=time.perf_counter()
 try:
  verify_module(t,wasm_hash)
  # Reference contains expected tensors, but only token IDs enter graph.forward.
  hidden=graph.forward(token_ids,args.layers)
  report=dict(scope='Client prefix preparation' if args.prepare_prefix else 'Client prefix continuation; preparation excluded',**session,tokens=len(token_ids),layers=graph.layers,query_cache_controlled=False,communication_scope='Candid request+reply; HTTP/CBOR/signatures excluded',instruction_scope='Handler counter excludes CDK Candid decode/encode',weight_precision=('Base per-row INT8, scalar/norm BF16/F32, separate F32 LoRA/readout' if any(w['dtype']=='int8' for w in m['tensors']) else 'Original BF16/F32 + separate F32 LoRA'),activation_precision=('INT8 block256 transport at activation query boundaries; F32 recurrent state and delta gates preserved; BF16/F32 numerical kernels' if args.wire_codec=='int8-block256-v1' else 'BF16 at official boundaries, F32 recurrent state; no added quantization'),queries=t.measurements,replayed_queries=t.replayed)
  if args.arithmetic=='int8':report['activation_precision']='Per-token block256 INT8 activation only inside base integer projections; F32 input preserved for LoRA, recurrent states and readout; lossless wire'
  if args.layers==32 and not args.prepare_prefix:
   np.save(directory/'final-hidden.npy',hidden)
   h=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],input_hash=input_hash,step=t.index,op='matmul',tensor='readout-f32',dims=[1,256,2560],scalars=[]);state=directory/'decision.request.bin'
   if args.wire_codec=='int8-block256-v1':h['encoding']=args.wire_codec
   atomic(state,encode(h,hidden[-1]));decision=t.command(dict(op='decision',method='decision_fast',input=str(state),options=record['options']));report['decision_query']=decision
   result=decision['ok']['decision']
   report['comparison']=dict(value=result['value'],typed_output_valid=bool(len(result['probabilities'])==len(record['options']) and (result['value'] is None or result['value'] in record['options']) and result['abstained']==(result['value'] is None) and abs(sum(result['probabilities'])+result['unknown_probability']-1)<1e-5),gold=record.get('gold'))
   if all(k in record for k in ('raw_result','result','hidden')):
    expected=np.array(list(record['raw_result']['raw_logits'].values()),dtype=np.float32)
    report['comparison'].update(reference_value=record['result']['value'],matches_reference=result['value']==record['result']['value'],hidden_max_error=float(abs(hidden[-1]-np.asarray(record['hidden'],dtype=np.float32)).max()),logit_max_error=float(abs(np.asarray(result['raw_logits'])-expected).max()))
   print('decision',json.dumps(report['comparison']),flush=True)
  report['wall_seconds_this_run']=time.perf_counter()-start;report['query_count']=len(t.measurements)+(args.layers==32 and not args.prepare_prefix);report['total_instructions']=sum(q['ok']['instructions'] for q in t.measurements)+(report.get('decision_query',{}).get('ok',{}).get('decision',{}).get('instructions',0));report['total_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in t.measurements)+sum(report.get('decision_query',{}).get('ok',{}).get(k,0) for k in ['request_bytes','reply_bytes']);report['max_query_instructions']=max(q['ok']['instructions'] for q in t.measurements);report['max_observed_heap_bytes']=max(q['ok']['heap_pages'] for q in t.measurements)*65536
  executed=[q for q in t.measurements if not q.get('replayed')]
  report['executed_query_count']=len(executed)+(args.layers==32 and not args.prepare_prefix);report['executed_instructions']=sum(q['ok']['instructions'] for q in executed)+report.get('decision_query',{}).get('ok',{}).get('decision',{}).get('instructions',0);report['executed_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in executed)+sum(report.get('decision_query',{}).get('ok',{}).get(k,0) for k in ['request_bytes','reply_bytes'])
  report['wasm_sha256']=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest();report['implementation_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['client/full_inference.py','client/transport.py','crates/imajev-runtime/src/lib.rs','canisters/inference/src/lib.rs']}
  report['processed_tokens']=len(token_ids) if args.prepare_prefix else len(token_ids)-len(cache['metadata']['token_ids'])
  report['prefix_preparation_included']=bool(args.prepare_prefix)
  report['prefix_cache_load_seconds']=cache_load_seconds
  report['implementation_hashes'].update({p:file_hash(ROOT/p) for p in ['client/prefix_inference.py','crates/imajev-runtime/src/int8_kernel.rs','scripts/run_prefix_canister.py']})
  verify_module(t,wasm_hash)
  if args.prepare_prefix:
   files={f'layer-{i:02d}.npy' for i in range(32)}|{f'states/layer-{i:02d}.npz' for i in range(32)}
   cm=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],wasm_sha256=wasm_hash,graph_sha256=graph_hash(),token_ids=token_ids,files={f:file_hash(t.directory/f) for f in sorted(files)})
   atomic(t.directory/'cache.json',(json.dumps(cm,indent=2)+'\n').encode())
  report['deployed_wasm_sha256']=wasm_hash
  report['end_to_end_seconds_excluding_process_startup']=time.perf_counter()-overall_start
  atomic(directory/'report.json',(json.dumps(report,indent=2)+'\n').encode());print('saved',directory/'report.json',flush=True)
 finally:t.close()
if __name__=='__main__':main()
