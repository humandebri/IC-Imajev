#!/usr/bin/env python3
"""Run from token IDs to dedicated decision using only ordinary canister queries."""
import argparse,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import JournalTransport,TextGraph,delta_full_log_enabled
from prefix_inference import PrefixTextGraph,load_cache,file_hash,graph_hash,verify_module,saved_delta_state_version
from decision_validation import validate_decision
from transport import encode,atomic,Transport

def main():
 overall_start=time.perf_counter()
 ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--step-offset',type=int,default=0);ap.add_argument('--query32-start',action='store_true');ap.add_argument('--packed-start',action='store_true');ap.add_argument('--bridge-binary');ap.add_argument('--adaptive-start',action='store_true');ap.add_argument('--tail-start',action='store_true');ap.add_argument('--tail-heads28',type=int,default=6);ap.add_argument('--tail-front28',type=int,default=5888);ap.add_argument('--tail-heads29',type=int,default=24);ap.add_argument('--tail-front30',type=int,default=512);ap.add_argument('--tail-down29',type=int,default=1024);ap.add_argument('--roll-start',action='store_true');ap.add_argument('--join-start',action='store_true');ap.add_argument('--roll-begin',type=int,default=4096);ap.add_argument('--roll-heads',type=int,default=18);ap.add_argument('--roll-down',type=int,default=768);ap.add_argument('--wasm',default='target/wasm32-unknown-unknown/release/imajev_inference.wasm');ap.add_argument('--url',default='http://localhost:8001/');ap.add_argument('--canister',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--prepare-prefix',action='store_true');ap.add_argument('--hybrid-cache');ap.add_argument('--prefix-tokens',type=int,help='Defaults to the input record prefix_tokens, or 45 for legacy fixtures');ap.add_argument('--manifest',default='checkpoints/full-int8.manifest.json');ap.add_argument('--reference',default='artifacts/reference-first.json');ap.add_argument('--record',type=int,default=0);ap.add_argument('--directory',default='artifacts/full-int8-efficient-run');ap.add_argument('--layers',type=int,default=32);ap.add_argument('--wire-codec',choices=['f32','bf16-exact','bf16-block256-exact-v1','int8-block256-v1'],default='bf16-exact');ap.add_argument('--arithmetic',choices=['f32','int8'],default='f32');ap.add_argument('--compact-lossless',action='store_true');ap.add_argument('--fuse-add-norm',action='store_true');ap.add_argument('--fuse-mlp',action='store_true');ap.add_argument('--fuse-mlp-norm',action='store_true');ap.add_argument('--fuse-norm-rope',action='store_true');ap.add_argument('--group-projections',action='store_true');ap.add_argument('--projection-group-cap',type=int,choices=[1,2,3],default=2);ap.add_argument('--attention-head-cap',type=int,choices=[1,2,4,8],default=1);ap.add_argument('--delta-head-cap',type=int,choices=[1,2,4,8,16],default=1);ap.add_argument('--row-cap',type=int,default=1536);ap.add_argument('--work-cap',type=int,default=450000000);ap.add_argument('--token-cap',type=int);ap.add_argument('--compact-heads',action='store_true');ap.add_argument('--fuse-delta',action='store_true');ap.add_argument('--wide-mlp',action='store_true');ap.add_argument('--terminal-readout',action='store_true');ap.add_argument('--fuse-delta-input',action='store_true');ap.add_argument('--reuse-projection-inputs',action='store_true');ap.add_argument('--frame-checksum',choices=['sha256','blake3','host'],default='sha256');ap.add_argument('--fuse-attention',action='store_true');ap.add_argument('--fuse-delta-projected',action='store_true');ap.add_argument('--fuse-delta-finish',action='store_true');ap.add_argument('--fuse-mlp-pipeline',action='store_true');ap.add_argument('--fuse-attention-full',action='store_true');ap.add_argument('--fuse-terminal-attention',action='store_true');ap.add_argument('--fuse-terminal-tail',action='store_true');ap.add_argument('--fuse-terminal-decision',action='store_true');ap.add_argument('--fuse-prefix-start',action='store_true');ap.add_argument('--fuse-mlp-full',action='store_true');ap.add_argument('--fuse-delta-full-log',action='store_true');ap.add_argument('--mlp-full-token-cap',type=int,choices=[87,89],default=87);args=ap.parse_args()
 if not args.tail_start and (args.tail_heads28,args.tail_front28,args.tail_heads29,args.tail_front30,args.tail_down29)!=(6,5888,24,512,1024):raise SystemExit('tail chunk overrides require --tail-start')
 if (args.adaptive_start or args.packed_start) and not args.tail_start:raise SystemExit('adaptive start requires tail route for fallback')
 if args.tail_start:
  if not args.join_start or not args.roll_start or args.prepare_prefix:raise SystemExit('tail start requires joined/rolled suffix mode')
  from tail_inference import validate_tail_chunks
  try:validate_tail_chunks(args.tail_heads28,args.tail_front28,args.tail_heads29,args.tail_front30,args.tail_down29)
  except ValueError as error:raise SystemExit(str(error))
 if args.token_cap is None:args.token_cap=512 if args.arithmetic=='int8' else 132
 m=json.loads((ROOT/args.manifest).read_text());fixture=json.loads((ROOT/args.reference).read_text());record=fixture['records'][args.record];token_ids=record['token_ids'];input_hash=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
 if not 1<=args.layers<=32:raise SystemExit('layers must be 1..32')
 if fixture['model_lock_sha256']!=m['model']:raise SystemExit('reference/model lock mismatch')
 if input_hash!=record['input_sha256']:raise SystemExit('token identity mismatch')
 if args.layers!=32 or args.arithmetic!='int8' or args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1'):raise SystemExit('prefix path requires full integer/lossless graph')
 if args.prepare_prefix:
  if args.prefix_tokens is None:args.prefix_tokens=record.get('prefix_tokens',45)
  if not 1<=args.prefix_tokens<len(token_ids):raise SystemExit('prefix token count')
  token_ids=token_ids[:args.prefix_tokens];input_hash=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
 wasm_hash=file_hash(ROOT/args.wasm)
 cache_module=wasm_hash
 if args.query32_start:
  import query32_balanced
  verified_modules={
   '6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4',
   'f6b399b6adb4e30871fef9833b89c766f099dc6cf97f736bd4e8f98464d9cf0b',
   'b23e62889cae4c5502457f5368c8f197db8ba16173420b6bf5b74d0523633b8b',
   '2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05',
  }
  if wasm_hash not in verified_modules:raise SystemExit('query32 requires a verified candidate module')
  # The original graph source remains frozen for prior proof identities.
  # Planning metadata names the selected, separately verified deployment.
  query32_balanced.MODULE=wasm_hash
  if args.prepare_prefix or not args.tail_start or not args.join_start or not args.roll_start or not args.bridge_binary or args.packed_start or args.adaptive_start:raise SystemExit('query32 requires tail/join/roll and its bridge; excludes other adaptive modes')
  if file_hash(ROOT/args.bridge_binary)!='eb7589996716bcd09f57ffd65ef6c9eff07f12bc4f8a883cd5f114c92690e486':raise SystemExit('query32 bridge identity mismatch')
  # Reuse only the original prefix module whose hidden/state bits were verified.
  cache_module='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
  if m['pack_hash']=='dd00c307ff65f2ae03ce5ee935d1dcfc7e298b6bd496e0aae2be1d81cd04cad6':cache_module='6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'
 verifier=Transport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/module-verification',m['pack_hash'])
 try:verify_module(verifier,wasm_hash)
 finally:verifier.close()
 requested_prefix_cache=args.cache;prefix_bank='common'
 if args.query32_start:prefix_bank='explicit-experiment-cache'
 cache_start=time.perf_counter()
 cache=None if args.prepare_prefix else load_cache(ROOT/args.cache,m,cache_module)
 if args.hybrid_cache:
  if args.prepare_prefix or not args.compact_heads or not args.fuse_delta_full_log:raise SystemExit('hybrid prefix requires suffix, compact heads and full Delta')
  from prefix_hybrid import load_packets
  cache['hybrid_packets']=load_packets(ROOT/args.hybrid_cache,cache)
 if args.query32_start:
  allowed_prefixes=(27,38) if wasm_hash in ('2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05','6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4') else (27,)
  p=len(cache['metadata']['token_ids'])
  if p not in allowed_prefixes or not 1<=len(token_ids)-p<=59:raise SystemExit('query32 prefix/suffix outside verified route bounds')
 cache_load_seconds=time.perf_counter()-cache_start
 directory=ROOT/args.directory;directory.mkdir(parents=True,exist_ok=True)
 session=dict(version=1,graph='qwen35-text-bf16-v1',model=m['model'],pack_hash=m['pack_hash'],input_hash=input_hash,url=args.url,canister=args.canister,row_cap=args.row_cap,token_cap=args.token_cap,work_cap=args.work_cap,wire_codec=args.wire_codec)
 effective_log=delta_full_log_enabled(args.fuse_delta_full_log,len(token_ids) if args.prepare_prefix else len(token_ids)-len(cache['metadata']['token_ids']),args.wire_codec,args.prepare_prefix or not args.compact_heads,None if args.prepare_prefix else cache['metadata']['version'])
 if args.hybrid_cache:session['hybrid_prefix_identity']=file_hash(ROOT/args.hybrid_cache/'cache.json')
 session['delta_full_log_requested']=args.fuse_delta_full_log
 session['delta_full_log_effective']=effective_log
 session['delta_state_version']=(2 if effective_log else 1) if args.prepare_prefix else cache['metadata']['version']
 if args.arithmetic=='int8':
  if args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1') or args.group_projections:raise SystemExit('integer evaluation requires bf16-exact, without group-projections')
  session["arithmetic"]="int8-block256-base-f32-lora-v1"
  session["integer_scheduler"]="aligned-token8-v1"
 if args.fuse_mlp:
  if args.arithmetic!='int8' or args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1'):raise SystemExit('fuse-mlp requires int8 arithmetic and lossless wire')
  session['fuse_mlp']='gate-up-swiglu-v1'
 if args.fuse_norm_rope:
  if args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1'):raise SystemExit('norm/RoPE fusion requires lossless wire')
  session['fuse_norm_rope']='rms-rope-shared-angles-v1'
 if args.fuse_mlp_norm:
  if not args.fuse_mlp or not args.fuse_add_norm:raise SystemExit('fuse-mlp-norm requires fused MLP and add/norm')
  session['fuse_mlp_norm']='residual-norm-mlp-v1'
  session['mlp_scheduler']='work-budget-v2'
 if args.fuse_add_norm:
  if args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1'):raise SystemExit('fuse-add-norm requires lossless wire')
  session['fuse_add_norm']=True
 if args.compact_lossless:
  if args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1'):raise SystemExit('compact-lossless requires bf16-exact wire')
  session["compact_lossless"]=True
 if args.group_projections:session["group_projections"]=True
 if args.projection_group_cap!=2:
  if not args.group_projections:raise SystemExit('projection-group-cap requires group-projections')
  session["projection_group_cap"]=args.projection_group_cap
 if args.compact_heads:
  session['compact_heads']='gqa-shared-terminal-delta-v1'
  session['terminal_states_retained']=args.prepare_prefix
 if args.reuse_projection_inputs:session['reuse_projection_inputs']='client-held-int8-f32-a-and-mlp-v2'
 if args.fuse_delta_full_log:session['fuse_delta_full_log']='full32-exact-prefix-innovation-v1'
 if args.fuse_delta_finish:session['fuse_delta_finish']='second-heads-output-projection-v1'
 if args.fuse_mlp_full:session['fuse_mlp_full']='complete-mlp-int8-v1'
 if args.mlp_full_token_cap!=87:session['mlp_full_token_cap']=args.mlp_full_token_cap
 if args.fuse_attention_full:session['fuse_attention_full']='complete-attention-int8-v1'
 if args.fuse_mlp_pipeline:session['fuse_mlp_pipeline']='prepared-int8-down-norm-v1'
 if args.fuse_delta_projected:session['fuse_delta_projected']='shared-integer-a-gates-stage-v1'
 if args.fuse_attention:session['fuse_attention']='integer-kv-q-gqa-gate-v1'
 session['terminal_readout']=bool(args.terminal_readout and not args.prepare_prefix)
 session['fuse_terminal_tail']=bool(args.fuse_terminal_tail and not args.prepare_prefix)
 if args.fuse_terminal_tail and (not args.fuse_terminal_attention or not args.fuse_terminal_decision or not args.fuse_mlp_full or args.layers!=32):raise SystemExit('terminal tail requires terminal decision/attention, full MLP and 32 layers')
 session['fuse_terminal_attention']=bool(args.fuse_terminal_attention and not args.prepare_prefix)
 session['fuse_terminal_decision']=bool(args.fuse_terminal_decision and not args.prepare_prefix)
 if session['fuse_terminal_decision'] and (not args.fuse_terminal_attention or args.layers!=32):raise SystemExit('terminal decision requires final fused attention and 32 layers')
 if args.fuse_prefix_start and (args.prepare_prefix or not args.hybrid_cache):raise SystemExit('prefix start requires hybrid suffix mode')
 session['fuse_prefix_start']=args.fuse_prefix_start
 if args.fuse_delta_input:session['fuse_delta_input']='qkv-gates-v1'
 if args.fuse_delta:session['fuse_delta']='projection-gates-conv-recurrence-v2'
 if args.wide_mlp:session['wide_mlp']='short-token89-4.5G-v2'
 if args.delta_head_cap>1:session["delta_head_cap"]=args.delta_head_cap
 if args.attention_head_cap>1:session["attention_head_cap"]=args.attention_head_cap
 session['prefix_mode']='prepare' if args.prepare_prefix else 'continue'
 session['prefix_identity']=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest() if args.prepare_prefix else cache['identity']
 session['prefix_graph_sha256']=graph_hash()
 session['wasm_sha256']=wasm_hash
 if args.query32_start:session.update(query32_start=True,query32_source_sha256=file_hash(ROOT/('client/query32_templates.py' if wasm_hash in ('2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05','6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4') else 'client/query32_balanced.py')),prefix_bank=prefix_bank,requested_prefix_cache=requested_prefix_cache,selected_prefix_cache=args.cache,selected_prefix_packets=args.hybrid_cache,prefix_source_module=cache_module,bridge_binary_sha256=file_hash(ROOT/args.bridge_binary))
 if args.adaptive_start:session.update(adaptive_start=True,adaptive_source_sha256=file_hash(ROOT/'client/adaptive_inference.py'))
 if args.packed_start:
  if not args.bridge_binary:raise SystemExit('packed start requires a bridge binary with mlp_delta_front support')
  session.update(packed_start=True,packed_source_sha256=file_hash(ROOT/'client/packed_inference.py'),bridge_binary_sha256=file_hash(ROOT/args.bridge_binary))
 if args.tail_start:session.update(tail_start=True,tail_source_sha256=file_hash(ROOT/'client/tail_inference.py'),tail_heads28=args.tail_heads28,tail_front28=args.tail_front28,tail_heads29=args.tail_heads29,tail_front30=args.tail_front30,tail_down29=args.tail_down29)
 if args.join_start and not args.roll_start:raise SystemExit('joined start requires rolled start')
 if args.join_start:session.update(join_start=True,join_source_sha256=file_hash(ROOT/'client/joined_inference.py'))
 session['roll_start']=bool(args.roll_start)
 if args.roll_start:session.update(roll_begin=args.roll_begin,roll_heads=args.roll_heads,roll_down=args.roll_down,roll_source_sha256=file_hash(ROOT/'client/roll_inference.py'))
 if args.frame_checksum!='sha256':session['frame_checksum']=args.frame_checksum
 if args.prepare_prefix and (ROOT/args.cache).resolve()!=(directory/'queries').resolve():raise SystemExit('prepare cache must equal directory/queries')
 if not 0<=args.step_offset<2**63:raise ValueError("step offset bounds")
 session["request_step_offset"]=args.step_offset
 path=directory/'session.json'
 if path.exists() and json.loads(path.read_text())!=session:raise SystemExit('Session mismatch; use a fresh directory')
 atomic(path,(json.dumps(session,indent=2)+'\n').encode())
 t=JournalTransport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),directory/'queries',m['pack_hash'],input_hash=input_hash,wire_codec='' if args.wire_codec=='f32' else args.wire_codec,frame_version={'sha256':1,'blake3':2,'host':3}[args.frame_checksum],bridge_binary=(ROOT/args.bridge_binary if args.bridge_binary else None))
 t.index=args.step_offset
 if args.reuse_projection_inputs:
  if args.arithmetic!='int8' or args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1'):raise ValueError('projection reuse requires integer lossless graph')
 t.reuse_projection_inputs=args.reuse_projection_inputs
 t.group_projections=args.group_projections
 t.projection_group_cap=args.projection_group_cap
 t.fuse_terminal_decision=session['fuse_terminal_decision'];t.decision_options=list(record['options'])
 if args.roll_start:
  if args.prepare_prefix or not args.hybrid_cache or args.layers!=32:raise SystemExit('rolled start requires prepared hybrid suffix mode')
  from roll_inference import RollPrefixGraph
  if args.join_start:
   from joined_inference import JoinedPrefixGraph
   RollPrefixGraph=JoinedPrefixGraph
 if args.tail_start:
  from tail_inference import TailPrefixGraph
  RollPrefixGraph=TailPrefixGraph
  if args.adaptive_start:
   from adaptive_inference import AdaptivePrefixGraph
   RollPrefixGraph=AdaptivePrefixGraph
  if args.packed_start:
   from packed_inference import PackedPrefixGraph
   RollPrefixGraph=PackedPrefixGraph
  if args.query32_start:
   if wasm_hash in ('2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05','6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'):
    import query32_templates
    query32_templates.MODULE=wasm_hash
    from query32_templates import Query32PrefixGraph
   else:
    from query32_balanced import Query32PrefixGraph
   RollPrefixGraph=Query32PrefixGraph
 graph=(TextGraph if args.prepare_prefix else (RollPrefixGraph if args.roll_start else PrefixTextGraph))(t,m,**({} if args.prepare_prefix else dict(cache=cache,fuse_prefix_start=args.fuse_prefix_start,**(dict(roll_begin=args.roll_begin,roll_heads=args.roll_heads,roll_down=args.roll_down,**(dict(tail_heads28=args.tail_heads28,tail_front28=args.tail_front28,tail_heads29=args.tail_heads29,tail_front30=args.tail_front30,tail_down29=args.tail_down29)if args.tail_start else{}))if args.roll_start else {}))),row_cap=args.row_cap,token_cap=args.token_cap,work_cap=args.work_cap,delta_head_cap=args.delta_head_cap,attention_head_cap=args.attention_head_cap,compact_lossless=args.compact_lossless,arithmetic=args.arithmetic,fuse_add_norm=args.fuse_add_norm,fuse_mlp=args.fuse_mlp,fuse_mlp_norm=args.fuse_mlp_norm,fuse_norm_rope=args.fuse_norm_rope,fuse_attention=args.fuse_attention,fuse_delta_projected=args.fuse_delta_projected,fuse_delta_finish=args.fuse_delta_finish,fuse_mlp_pipeline=args.fuse_mlp_pipeline,fuse_attention_full=args.fuse_attention_full,fuse_terminal_attention=(args.fuse_terminal_attention and not args.prepare_prefix),fuse_mlp_full=args.fuse_mlp_full,mlp_full_token_cap=args.mlp_full_token_cap,fuse_terminal_tail=session['fuse_terminal_tail'],fuse_delta_full_log=args.fuse_delta_full_log,compact_heads=args.compact_heads,terminal_readout=(args.terminal_readout and not args.prepare_prefix),fuse_delta_input=args.fuse_delta_input,fuse_delta=args.fuse_delta,wide_mlp=args.wide_mlp,retain_terminal_state=(args.prepare_prefix or not args.compact_heads));start=time.perf_counter()
 try:
  verify_module(t,wasm_hash)
  # Reference contains expected tensors, but only token IDs enter graph.forward.
  from limit_fallback import eligible,restart
  if args.tail_start and not args.query32_start and (directory/'fallback.json').exists():
   restart(directory,t,sys.argv[1:],primary_seconds=time.perf_counter()-start);verify_module(t,wasm_hash);print('saved fallback',directory/'report.json',flush=True);return
  try:hidden=graph.forward(token_ids,args.layers)
  except (ValueError,RuntimeError) as error:
   if args.query32_start or not args.tail_start or not eligible(error):raise
   print('limit reached; restarting standard queries in separate journal',flush=True)
   restart(directory,t,sys.argv[1:],error,primary_seconds=time.perf_counter()-start);verify_module(t,wasm_hash);print('saved fallback',directory/'report.json',flush=True);return
  report=dict(scope='Client prefix preparation' if args.prepare_prefix else 'Client prefix continuation; preparation excluded',**session,tokens=len(token_ids),layers=graph.layers,query_cache_controlled=False,communication_scope='Candid request+reply; HTTP/CBOR/signatures excluded',instruction_scope='Handler counter excludes CDK Candid decode/encode',weight_precision=('Base per-row INT8, scalar/norm BF16/F32, separate F32 LoRA/readout' if any(w['dtype']=='int8' for w in m['tensors']) else 'Original BF16/F32 + separate F32 LoRA'),activation_precision=('INT8 block256 transport at activation query boundaries; F32 recurrent state and delta gates preserved; BF16/F32 numerical kernels' if args.wire_codec=='int8-block256-v1' else 'BF16 at official boundaries, F32 recurrent state; no added quantization'),queries=t.measurements,replayed_queries=t.replayed)
  if args.query32_start:report['query32_effective']=graph.adaptive_effective
  if args.roll_start:report['roll_effective']=graph.roll_effective
  if args.join_start:report['join_effective']=graph.join_effective
  if args.adaptive_start:report['adaptive_effective']=graph.adaptive_effective
  if args.packed_start:report['packed_effective']=graph.adaptive_effective
  if args.tail_start:report['tail_effective']=graph.tail_effective
  if args.arithmetic=='int8':report['activation_precision']='Per-token block256 INT8 activation only inside base integer projections; F32 input preserved for LoRA, recurrent states and readout; lossless wire'
  if args.layers==32 and not args.prepare_prefix:
   np.save(directory/'final-hidden.npy',hidden)
   h=dict(version=t.frame_version,model=m['model'],pack_hash=m['pack_hash'],input_hash=input_hash,step=t.index,op='matmul',tensor='readout-f32',dims=[1,256,2560],scalars=[]);state=directory/'decision.request.bin'
   if args.wire_codec=='int8-block256-v1':h['encoding']=args.wire_codec
   if t.fuse_terminal_decision:
    decision=dict(ok=dict(decision=t.terminal_decision,request_bytes=0,reply_bytes=0),included_in_terminal_query=True)
   else:
    atomic(state,encode(h,hidden[-1]));decision=t.command(dict(op='decision',method='decision_fast',input=str(state),options=record['options']))
   report['decision_query']=decision
   result=validate_decision(decision['ok']['decision'],list(record['options']))
   report['comparison']=dict(value=result['value'],typed_output_valid=bool(len(result['probabilities'])==len(record['options']) and (result['value'] is None or result['value'] in record['options']) and result['abstained']==(result['value'] is None) and abs(sum(result['probabilities'])+result['unknown_probability']-1)<1e-5),gold=record.get('gold'))
   if all(k in record for k in ('raw_result','result','hidden')):
    expected=np.array(list(record['raw_result']['raw_logits'].values()),dtype=np.float32)
    report['comparison'].update(reference_value=record['result']['value'],matches_reference=result['value']==record['result']['value'],hidden_max_error=float(abs(hidden[-1]-np.asarray(record['hidden'],dtype=np.float32)).max()),logit_max_error=float(abs(np.asarray(result['raw_logits'])-expected).max()))
   print('decision',json.dumps(report['comparison']),flush=True)
  report['wall_seconds_this_run']=time.perf_counter()-start;report['query_count']=len(t.measurements)+(args.layers==32 and not args.prepare_prefix and not t.fuse_terminal_decision);report['total_instructions']=sum(q['ok']['instructions'] for q in t.measurements)+((0 if t.fuse_terminal_decision else report.get('decision_query',{}).get('ok',{}).get('decision',{}).get('instructions',0)));report['total_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in t.measurements)+sum(report.get('decision_query',{}).get('ok',{}).get(k,0) for k in ['request_bytes','reply_bytes']);report['max_query_instructions']=max(q['ok']['instructions'] for q in t.measurements);report['max_observed_heap_bytes']=max(q['ok']['heap_pages'] for q in t.measurements)*65536
  executed=[q for q in t.measurements if not q.get('replayed')]
  report['executed_query_count']=len(executed)+(args.layers==32 and not args.prepare_prefix and not t.fuse_terminal_decision);report['executed_instructions']=sum(q['ok']['instructions'] for q in executed)+(0 if t.fuse_terminal_decision else report.get('decision_query',{}).get('ok',{}).get('decision',{}).get('instructions',0));report['executed_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in executed)+sum(report.get('decision_query',{}).get('ok',{}).get(k,0) for k in ['request_bytes','reply_bytes'])
  report['wasm_sha256']=hashlib.sha256((ROOT/args.wasm).read_bytes()).hexdigest();report['implementation_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['client/full_inference.py','client/transport.py','crates/imajev-runtime/src/lib.rs','crates/imajev-runtime/src/delta_stage.rs','crates/imajev-runtime/src/delta_simd.rs','crates/imajev-runtime/src/int8_kernel.rs','crates/imajev-runtime/src/terminal.rs','canisters/inference/src/lib.rs']}
  report['implementation_hashes'].update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'crates/imajev-runtime/src').glob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']})
  report['processed_tokens']=len(token_ids) if args.prepare_prefix else len(token_ids)-len(cache['metadata']['token_ids'])
  report['prefix_preparation_included']=bool(args.prepare_prefix)
  report['prefix_cache_load_seconds']=cache_load_seconds
  report['implementation_hashes'].update({p:file_hash(ROOT/p) for p in ['client/prefix_inference.py','crates/imajev-runtime/src/int8_kernel.rs','scripts/run_prefix_canister.py']})
  verify_module(t,wasm_hash)
  if args.prepare_prefix:
   files={f'layer-{i:02d}.npy' for i in range(32)}|{f'states/layer-{i:02d}.npz' for i in range(32)}
   state_version=saved_delta_state_version(t.directory)
   if state_version!=session['delta_state_version']:raise ValueError('prefix Delta representation version')
   cm=dict(version=state_version,model=m['model'],pack_hash=m['pack_hash'],wasm_sha256=wasm_hash,graph_sha256=graph_hash(),token_ids=token_ids,files={f:file_hash(t.directory/f) for f in sorted(files)})
   atomic(t.directory/'cache.json',(json.dumps(cm,indent=2)+'\n').encode())
  report['deployed_wasm_sha256']=wasm_hash
  report['end_to_end_seconds_excluding_process_startup']=time.perf_counter()-overall_start
  atomic(directory/'report.json',(json.dumps(report,indent=2)+'\n').encode());print('saved',directory/'report.json',flush=True)
 finally:t.close()
if __name__=='__main__':main()
