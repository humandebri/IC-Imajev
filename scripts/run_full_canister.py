#!/usr/bin/env python3
"""Run from token IDs to dedicated decision using only ordinary canister queries."""
import argparse,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from full_inference import JournalTransport,TextGraph,delta_full_log_enabled
from decision_validation import validate_decision
from transport import encode,atomic
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wasm',default='target/wasm32-unknown-unknown/release/imajev_inference.wasm');ap.add_argument('--url',default='http://localhost:8001/');ap.add_argument('--canister',default='46el7-ql777-77775-aaada-cai');ap.add_argument('--manifest',default='checkpoints/full-int8.manifest.json');ap.add_argument('--reference',default='artifacts/reference-first.json');ap.add_argument('--record',type=int,default=0);ap.add_argument('--directory',default='artifacts/full-int8-efficient-run');ap.add_argument('--layers',type=int,default=32);ap.add_argument('--wire-codec',choices=['f32','bf16-exact','bf16-block256-exact-v1','int8-block256-v1'],default='bf16-exact');ap.add_argument('--arithmetic',choices=['f32','int8'],default='f32');ap.add_argument('--compact-lossless',action='store_true');ap.add_argument('--fuse-add-norm',action='store_true');ap.add_argument('--fuse-mlp',action='store_true');ap.add_argument('--fuse-mlp-norm',action='store_true');ap.add_argument('--fuse-norm-rope',action='store_true');ap.add_argument('--group-projections',action='store_true');ap.add_argument('--projection-group-cap',type=int,choices=[1,2,3],default=2);ap.add_argument('--attention-head-cap',type=int,choices=[1,2,4,8],default=1);ap.add_argument('--delta-head-cap',type=int,choices=[1,2,4,8,16],default=1);ap.add_argument('--row-cap',type=int,default=1536);ap.add_argument('--work-cap',type=int,default=450000000);ap.add_argument('--token-cap',type=int);ap.add_argument('--compact-heads',action='store_true');ap.add_argument('--fuse-delta',action='store_true');ap.add_argument('--wide-mlp',action='store_true');ap.add_argument('--terminal-readout',action='store_true');ap.add_argument('--fuse-delta-input',action='store_true');ap.add_argument('--reuse-projection-inputs',action='store_true');ap.add_argument('--frame-checksum',choices=['sha256','blake3','host'],default='sha256');ap.add_argument('--fuse-attention',action='store_true');ap.add_argument('--fuse-delta-projected',action='store_true');ap.add_argument('--fuse-delta-finish',action='store_true');ap.add_argument('--fuse-mlp-pipeline',action='store_true');ap.add_argument('--fuse-attention-full',action='store_true');ap.add_argument('--fuse-terminal-attention',action='store_true');ap.add_argument('--fuse-terminal-tail',action='store_true');ap.add_argument('--fuse-terminal-decision',action='store_true');ap.add_argument('--fuse-mlp-full',action='store_true');ap.add_argument('--fuse-delta-full-log',action='store_true');ap.add_argument('--mlp-full-token-cap',type=int,choices=[87,89],default=87);ap.add_argument("--step-offset",type=int,default=0,help="Unique request step range for repeated measurements");args=ap.parse_args()
 if not 0<=args.step_offset<2**63:raise ValueError('step offset bounds')
 if args.token_cap is None:args.token_cap=512 if args.arithmetic=='int8' else 132
 m=json.loads((ROOT/args.manifest).read_text());fixture=json.loads((ROOT/args.reference).read_text());record=fixture['records'][args.record];token_ids=record['token_ids'];input_hash=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
 if not 1<=args.layers<=32:raise SystemExit('layers must be 1..32')
 if fixture['model_lock_sha256']!=m['model']:raise SystemExit('reference/model lock mismatch')
 if input_hash!=record['input_sha256']:raise SystemExit('token identity mismatch')
 directory=ROOT/args.directory;directory.mkdir(parents=True,exist_ok=True)
 wasm_hash=hashlib.sha256((ROOT/args.wasm).read_bytes()).hexdigest()
 session=dict(version=1,graph='qwen35-text-bf16-v1',model=m['model'],pack_hash=m['pack_hash'],input_hash=input_hash,url=args.url,canister=args.canister,row_cap=args.row_cap,token_cap=args.token_cap,work_cap=args.work_cap,wire_codec=args.wire_codec)
 session['delta_full_log_requested']=args.fuse_delta_full_log
 session['delta_full_log_effective']=delta_full_log_enabled(args.fuse_delta_full_log,len(token_ids),args.wire_codec,not args.compact_heads)
 session['delta_state_version']=(2 if session['delta_full_log_effective'] else 1) if not args.compact_heads else None
 session['wasm_sha256']=wasm_hash
 if args.step_offset:session['request_step_offset']=args.step_offset
 if args.frame_checksum!='sha256':session['frame_checksum']=args.frame_checksum
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
  session['terminal_states_retained']=False
 if args.reuse_projection_inputs:session['reuse_projection_inputs']='client-held-int8-f32-a-and-mlp-v2'
 if args.fuse_delta_full_log:session['fuse_delta_full_log']='full32-exact-prefix-innovation-v1'
 if args.fuse_delta_finish:session['fuse_delta_finish']='second-heads-output-projection-v1'
 if args.fuse_mlp_full:session['fuse_mlp_full']='complete-mlp-int8-v1'
 if args.mlp_full_token_cap!=87:session['mlp_full_token_cap']=args.mlp_full_token_cap
 if args.fuse_attention_full:session['fuse_attention_full']='complete-attention-int8-v1'
 if args.fuse_mlp_pipeline:session['fuse_mlp_pipeline']='prepared-int8-down-norm-v1'
 if args.fuse_delta_projected:session['fuse_delta_projected']='shared-integer-a-gates-stage-v1'
 if args.fuse_attention:session['fuse_attention']='integer-kv-q-gqa-gate-v1'
 session['terminal_readout']=bool(args.terminal_readout)
 session['fuse_terminal_tail']=bool(args.fuse_terminal_tail)
 if args.fuse_terminal_tail and (not args.fuse_terminal_attention or not args.fuse_terminal_decision or not args.fuse_mlp_full or args.layers!=32):raise SystemExit('terminal tail requires terminal decision/attention, full MLP and 32 layers')
 session['fuse_terminal_attention']=bool(args.fuse_terminal_attention)
 session['fuse_terminal_decision']=bool(args.fuse_terminal_decision)
 if session['fuse_terminal_decision'] and (not args.fuse_terminal_attention or args.layers!=32):raise SystemExit('terminal decision requires final fused attention and 32 layers')
 if args.fuse_delta_input:session['fuse_delta_input']='qkv-gates-v1'
 if args.fuse_delta:session['fuse_delta']='projection-gates-conv-recurrence-v2'
 if args.wide_mlp:session['wide_mlp']='short-token89-4.5G-v2'
 if args.delta_head_cap>1:session["delta_head_cap"]=args.delta_head_cap
 if args.attention_head_cap>1:session["attention_head_cap"]=args.attention_head_cap
 path=directory/'session.json'
 if path.exists() and json.loads(path.read_text())!=session:raise SystemExit('Session mismatch; use a fresh directory')
 atomic(path,(json.dumps(session,indent=2)+'\n').encode())
 t=JournalTransport(m['model'],args.url,args.canister,str(ROOT/'artifacts/imajev-local.pem'),directory/'queries',m['pack_hash'],input_hash=input_hash,wire_codec='' if args.wire_codec=='f32' else args.wire_codec,frame_version={'sha256':1,'blake3':2,'host':3}[args.frame_checksum])
 t.index=args.step_offset
 if args.reuse_projection_inputs:
  if args.arithmetic!='int8' or args.wire_codec not in ('bf16-exact','bf16-block256-exact-v1'):raise ValueError('projection reuse requires integer lossless graph')
 t.reuse_projection_inputs=args.reuse_projection_inputs
 t.group_projections=args.group_projections
 t.projection_group_cap=args.projection_group_cap
 t.fuse_terminal_decision=session['fuse_terminal_decision'];t.decision_options=list(record['options'])
 graph=TextGraph(t,m,row_cap=args.row_cap,token_cap=args.token_cap,work_cap=args.work_cap,delta_head_cap=args.delta_head_cap,attention_head_cap=args.attention_head_cap,compact_lossless=args.compact_lossless,arithmetic=args.arithmetic,fuse_add_norm=args.fuse_add_norm,fuse_mlp=args.fuse_mlp,fuse_mlp_norm=args.fuse_mlp_norm,fuse_norm_rope=args.fuse_norm_rope,fuse_attention=args.fuse_attention,fuse_delta_projected=args.fuse_delta_projected,fuse_delta_finish=args.fuse_delta_finish,fuse_mlp_pipeline=args.fuse_mlp_pipeline,fuse_attention_full=args.fuse_attention_full,fuse_terminal_attention=args.fuse_terminal_attention,fuse_mlp_full=args.fuse_mlp_full,mlp_full_token_cap=args.mlp_full_token_cap,fuse_terminal_tail=session['fuse_terminal_tail'],fuse_delta_full_log=args.fuse_delta_full_log,compact_heads=args.compact_heads,terminal_readout=(args.terminal_readout),fuse_delta_input=args.fuse_delta_input,fuse_delta=args.fuse_delta,wide_mlp=args.wide_mlp,retain_terminal_state=not args.compact_heads);start=time.perf_counter()
 try:
  verify_module(t,wasm_hash)
  # Reference contains expected tensors, but only token IDs enter graph.forward.
  hidden=graph.forward(token_ids,args.layers)
  report=dict(scope='Full text model query inference' if args.layers==32 else f'Partial {args.layers}-layer query validation',**session,tokens=len(token_ids),layers=graph.layers,query_cache_controlled=False,communication_scope='Candid request+reply; HTTP/CBOR/signatures excluded',instruction_scope='Handler counter excludes CDK Candid decode/encode',weight_precision=('Base per-row INT8, scalar/norm BF16/F32, separate F32 LoRA/readout' if any(w['dtype']=='int8' for w in m['tensors']) else 'Original BF16/F32 + separate F32 LoRA'),activation_precision=('INT8 block256 transport at activation query boundaries; F32 recurrent state and delta gates preserved; BF16/F32 numerical kernels' if args.wire_codec=='int8-block256-v1' else 'BF16 at official boundaries, F32 recurrent state; no added quantization'),queries=t.measurements,replayed_queries=t.replayed)
  if args.arithmetic=='int8':report['activation_precision']='Per-token block256 INT8 activation only inside base integer projections; F32 input preserved for LoRA, recurrent states and readout; lossless wire'
  if args.layers==32:
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
  report['wall_seconds_this_run']=time.perf_counter()-start;report['query_count']=len(t.measurements)+(args.layers==32 and not t.fuse_terminal_decision);report['total_instructions']=sum(q['ok']['instructions'] for q in t.measurements)+((0 if t.fuse_terminal_decision else report.get('decision_query',{}).get('ok',{}).get('decision',{}).get('instructions',0)));report['total_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in t.measurements)+sum(report.get('decision_query',{}).get('ok',{}).get(k,0) for k in ['request_bytes','reply_bytes']);report['max_query_instructions']=max(q['ok']['instructions'] for q in t.measurements);report['max_observed_heap_bytes']=max(q['ok']['heap_pages'] for q in t.measurements)*65536
  executed=[q for q in t.measurements if not q.get('replayed')]
  report['executed_query_count']=len(executed)+(args.layers==32 and not t.fuse_terminal_decision);report['executed_instructions']=sum(q['ok']['instructions'] for q in executed)+(0 if t.fuse_terminal_decision else report.get('decision_query',{}).get('ok',{}).get('decision',{}).get('instructions',0));report['executed_candid_bytes']=sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in executed)+sum(report.get('decision_query',{}).get('ok',{}).get(k,0) for k in ['request_bytes','reply_bytes'])
  report['wasm_sha256']=hashlib.sha256((ROOT/args.wasm).read_bytes()).hexdigest();report['implementation_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['client/full_inference.py','client/transport.py','crates/imajev-runtime/src/lib.rs','crates/imajev-runtime/src/delta_stage.rs','crates/imajev-runtime/src/delta_simd.rs','crates/imajev-runtime/src/int8_kernel.rs','crates/imajev-runtime/src/terminal.rs','canisters/inference/src/lib.rs']}
  report['implementation_hashes'].update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'crates/imajev-runtime/src').glob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']})
  report['deployed_wasm_sha256']=verify_module(t,wasm_hash)
  atomic(directory/'report.json',(json.dumps(report,indent=2)+'\n').encode());print('saved',directory/'report.json',flush=True)
 finally:t.close()
if __name__=='__main__':main()
