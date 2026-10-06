#!/usr/bin/env python3
"""Measure actual fused MLP/Delta boundary, with exact frozen state references."""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
from mlp_delta_stream_codec import NAME,encode_request,encode_continue_request
from mlp_stream_codec import NAME as MLP_NAME,C,H,R
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--continue-mlp',action='store_true')
ap.add_argument('--start-first',action='store_true')
ap.add_argument('--first-only',action='store_true')
ap.add_argument('--start-ids',action='store_true')
ap.add_argument('--compress-base',action='store_true')
ap.add_argument('--raw-finish',action='store_true')
ap.add_argument('--finish-next-delta',action='store_true')
ap.add_argument('--down-rows',type=int,default=0)
ap.add_argument('--layers',default='0,1,3')
ap.add_argument('--begins',default='4352,4608,4864,5120')
ap.add_argument('--heads',default='22,24')
ap.add_argument('--source-case',choices=['617','insufficient','maximum'],default='617')
a=ap.parse_args()
if a.finish_next_delta and (not a.down_rows or not a.continue_mlp):ap.error('--finish-next-delta requires --down-rows and --continue-mlp')
if a.start_ids and (not a.start_first or a.layers!='0'):ap.error('--start-ids requires --start-first --layers 0')
if a.down_rows and (not a.continue_mlp or not 0<a.down_rows<C or a.down_rows%32):ap.error('--down-rows requires --continue-mlp and 32-aligned rows in 1..2559')
d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();wasm=sha(ROOT/a.wasm)
m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes())
source=ROOT/'artifacts/prefix_codec/full-capture-prepared-proof'/a.source_case;report=json.loads((source/'report.json').read_bytes())
old=ROOT/f'artifacts/column16-v1-{a.source_case}';old_report=json.loads((old/'report.json').read_bytes())
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__),ROOT/'MODEL_LOCK.json',ROOT/'checkpoints/full-int8.manifest.json'];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths}
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec=MLP_NAME,frame_version=3)
cases=[]
try:
 verify_module(t,wasm)
 for layer in [int(v)for v in a.layers.split(',')]:
  q=next(q for q in report['queries']if q['op']=='mlp_full_integer'and f'.layers.{layer}.'in q['tensor'])
  req=source/'queries'/f"{q['index']:06d}.request.bin";res=source/'queries'/f"{q['index']:06d}.response.bin";head,x=decode(req.read_bytes());_,both=decode(res.read_bytes());n=head['dims'][0];next_layer=layer+1
  tensor=f'model.language_model.layers.{next_layer}.linear_attn.in_proj_qkv.weight'
  capture=next(q for q in old_report['queries']if q['op']=='delta_project_capture'and q['tensor']==tensor)
  creq=old/'queries'/f"{capture['index']:06d}.request.bin";cres=old/'queries'/f"{capture['index']:06d}.response.bin";ch,cx=decode(creq.read_bytes());_,cy=decode(cres.read_bytes())
  assert ch['dims']==[n,16,0,0] and cx[:n*C].tobytes()==both[n*C:].tobytes()
  expected_prepared=cy[n*2048+3*4096:];assert expected_prepared.size==n*2762
  outq=next(q for q in old_report['queries']if q['op']=='lora_integer'and q['tensor']==tensor.replace('in_proj_qkv','out_proj'))
  outreq=old/'queries'/f"{outq['index']:06d}.request.bin";_,all_gated=decode(outreq.read_bytes());all_gated=all_gated.reshape(n,4096)
  prefix=ROOT/'artifacts/output-pairs-v2-prefix/queries/states'/f'layer-{next_layer:02d}.npz'
  frozen_state=ROOT/f'artifacts/output-pairs-v2-{a.source_case}/queries/states'/f'layer-{next_layer:02d}.npz'
  with np.load(prefix,allow_pickle=False)as z:history=z['conv'].copy();log=z['delta_log'].copy();p=log.size//6176;assert p==45
  with np.load(frozen_state,allow_pickle=False)as z:final_history=z['conv'].copy()
  for begin in [int(v) for v in a.begins.split(',')]:
   start_call=None
   if a.start_first:
    from delta_mlp_start_codec import NAME as START_NAME,CONV,encode_request as encode_start
    sq=next(q for q in old_report['queries']if q['op']=='delta_project_capture'and q['tensor']==f'model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight')
    _,initial=decode((old/'queries'/f"{sq['index']:06d}.request.bin").read_bytes())
    with np.load(ROOT/'artifacts/output-pairs-v2-prefix/queries/states'/f'layer-{layer:02d}.npz',allow_pickle=False)as z:cv=z['conv'].copy();lv=z['delta_log'].copy()
    sh=dict(head,version=3,step=t.index,encoding=START_NAME,op='delta_mlp_stream_start_ids' if a.start_ids else 'delta_mlp_stream_prepare',dims=[n,begin,p])
    if a.start_ids:
     from delta_mlp_start_codec import encode_ids
     ref=json.loads((ROOT/'artifacts/reference-serving.json').read_text());assert ref['model_lock_sha256']==m['model'];ids=ref['records'][{'617':0,'maximum':11,'insufficient':19}[a.source_case]]['token_ids'][45:];assert len(ids)==n;packet=encode_ids(sh,ids,cv,lv)
    else:packet=encode_start(sh,x[:n*C],initial[:n*C],cv,lv)
    si=t.index
    try:sy=t._run_encoded(sh,packet)
    except RuntimeError as error:
     if not is_instruction_limit(error):raise
     failed=d/f'failed-start-layer{layer}-begin{begin}.request.bin';shutil.copyfile(d/f'{si:06d}.request.bin',failed)
     row=dict(layer=layer,tokens=n,begin=begin,success=False,stage='start_query',instruction_limit=True,error=str(error),failed_request_sha256=sha(failed));cases.append(row);print(json.dumps(row),flush=True);continue
    start_call=t.measurements[-1];state=sy[:-CONV]
    with np.load(ROOT/f'artifacts/output-pairs-v2-{a.source_case}/queries/states'/f'layer-{layer:02d}.npz',allow_pickle=False)as z:assert sy[-CONV:].tobytes()==z['conv'].ravel().tobytes()
    po=d/f'{si:06d}.profile.response.bin';start_profile=t.command(dict(op='profile',input=str(d/f'{si:06d}.request.bin'),output=str(po)));assert decode(po.read_bytes())[1].tobytes()==sy.tobytes()
   t.wire_codec=MLP_NAME;expected_start=t.run('mlp_stream_prepare',x,[n,0,begin],[2.,1e-6],tensor=head['tensor'],aux=head['aux'],input_hash=head['input_hash'])
   if a.start_first:assert state.tobytes()==expected_start.tobytes()
   else:state=expected_start
   if a.first_only:
    if not start_call:raise ValueError('first-only requires start-first')
    row=dict(layer=layer,tokens=n,begin=begin,stage='start_only',success=True,start_call=start_call,start_profile=start_profile,start_bitwise_equal=True,history_bitwise_equal=True,source_request_sha256=sha(req),source_response_sha256=sha(res));cases.append(row);print(json.dumps(row),flush=True);continue
   for heads in [int(v) for v in a.heads.split(',')]:
    cols=heads*128;indices=np.concatenate([np.arange(heads//2*128),np.arange(2048,2048+heads//2*128),np.arange(4096,4096+cols)])
    sliced=np.concatenate([log[:p*2048].reshape(p,2048)[:,:heads//2*128].ravel(),log[p*2048:p*6144].reshape(p,4096)[:,:cols].ravel(),log[p*6144:].reshape(p,32)[:,:heads].ravel()])
    h=dict(version=3,model=m['model'],pack_hash=m['pack_hash'],input_hash=head['input_hash'],step=t.index,op='mlp_complete_delta_partial',encoding=NAME,tensor=head['tensor'],aux=head['aux'],scalars=[2.,1e-6],dims=[n,begin,H-begin,heads,p])
    packet=encode_request(h,state,history[:,indices],sliced);index=t.index
    row=dict(layer=layer,next_layer=next_layer,tokens=n,begin=begin,count=H-begin,heads=heads,prefix_tokens=p,request_frame_bytes=len(packet),source_request_sha256=sha(req),source_response_sha256=sha(res),capture_request_sha256=sha(creq),capture_response_sha256=sha(cres),gated_reference_sha256=sha(outreq),prefix_state_sha256=sha(prefix),final_history_sha256=sha(frozen_state))
    if start_call:row.update(start_call=start_call,start_profile=start_profile,start_bitwise_equal=True)
    try:y=t._run_encoded(h,packet)
    except RuntimeError as error:
     if not is_instruction_limit(error):raise
     failed=d/f'failed-layer{layer}-begin{begin}-heads{heads}.request.bin';shutil.copyfile(d/f'{index:06d}.request.bin',failed)
     row.update(success=False,instruction_limit=True,error=str(error),failed_request_sha256=sha(failed));cases.append(row);print(json.dumps(row),flush=True);continue
    row['call']=t.measurements[-1]
    if start_call:row.update(start_call=start_call,start_profile=start_profile,start_bitwise_equal=True)
    assert y[:n*C].tobytes()==both[:n*C].tobytes()
    assert y[n*C:n*(C+2762)].tobytes()==expected_prepared.tobytes()
    assert y[-3*heads*256:].tobytes()==final_history[:,indices].ravel().tobytes()
    base=y[n*(C+2762):n*(2*C+2762)];ax=y[n*(2*C+2762):n*(2*C+2762+R)]
    # Verify both original-order out-projection carries independently using
    # the frozen full gated output, not host recurrence or normalization.
    op=tensor.replace('in_proj_qkv','out_proj');t.wire_codec='bf16-block256-exact-v1'
    expected_base=t.run('linear_integer_k_continue',np.concatenate([all_gated[:,:cols].ravel(),np.zeros(n*C,np.float32)]),[n,C,4096,0,cols],tensor=op,input_hash=head['input_hash'])
    expected_ax=t.run('matmul_k_continue',np.concatenate([all_gated[:,:cols].ravel(),np.zeros(n*R,np.float32)]),[n,R,4096,0,cols],tensor=op.replace('.weight','.lora_A.weight'),input_hash=head['input_hash'])
    assert base.tobytes()==expected_base.tobytes() and ax.tobytes()==expected_ax.tobytes()
    if a.continue_mlp:
     remaining=32-heads;rest_indices=np.concatenate([np.arange(heads//2*128,2048),np.arange(2048+heads//2*128,4096),np.arange(4096+cols,8192)])
     rest_log=np.concatenate([log[:p*2048].reshape(p,2048)[:,heads//2*128:].ravel(),log[p*2048:p*6144].reshape(p,4096)[:,cols:].ravel(),log[p*6144:].reshape(p,32)[:,heads:].ravel()])
     fh=dict(h,op='delta_partial_mlp_prepare_down' if a.down_rows else 'delta_partial_mlp_prepare',step=t.index,dims=h['dims']+([a.down_rows]if a.down_rows else []));follow_index=t.index
     try:follow_packet=encode_continue_request(fh,y,history[:,rest_indices],rest_log,compress_base=a.compress_base)
     except ValueError as error:
      if str(error)!='pair continuation frame bounds':raise
      row.update(success=False,stage='follow_frame',instruction_limit=False,frame_limit=True,error=str(error));cases.append(row);print(json.dumps(row),flush=True);continue
     row['follow_request_frame_bytes']=len(follow_packet)
     try:fy=t._run_encoded(fh,follow_packet)
     except RuntimeError as error:
      if not is_instruction_limit(error):raise
      failed=d/f'failed-follow-layer{layer}-begin{begin}-heads{heads}.request.bin';shutil.copyfile(d/f'{follow_index:06d}.request.bin',failed)
      row.update(success=False,stage='follow_query',instruction_limit=True,error=str(error),failed_request_sha256=sha(failed));cases.append(row);print(json.dumps(row),flush=True);continue
     row['follow_call']=t.measurements[-1]
     assert fy[-3*remaining*256:].tobytes()==final_history[:,rest_indices].ravel().tobytes()
     nextq=next(q for q in report['queries']if q['op']=='mlp_full_integer'and f'.layers.{next_layer}.'in q['tensor']);nextreq=source/'queries'/f"{nextq['index']:06d}.request.bin";nextres=source/'queries'/f"{nextq['index']:06d}.response.bin";nh,nx=decode(nextreq.read_bytes());_,next_expected=decode(nextres.read_bytes())
     t.wire_codec='mlp-down-state-exact-v1';prepared=t.run('mlp_prepare_partial_down' if a.down_rows else 'mlp_prepare_down',nx,[n,C]+([a.down_rows]if a.down_rows else []),[2.,1e-6],tensor=nh['tensor'],aux=nh['aux'],input_hash=nh['input_hash'])
     assert fy[:n*(C+H+100)].tobytes()==prepared.tobytes()
     if a.down_rows and a.raw_finish:
      finished=t.run('mlp_down_norm_partial_prepared',fy[:n*(C+H+100)],[n,C,a.down_rows],[2.,1e-6],tensor=nh['tensor'],aux=nh['aux'],input_hash=nh['input_hash'])
     elif a.down_rows:
      from mlp_delta_carry import NAME as CARRY_NAME,encode_request as encode_finish
      finish=dict(nh,version=3,step=t.index,encoding=CARRY_NAME,op='mlp_finish_partial_integer',dims=[n,0,a.down_rows]);finished=t._run_encoded(finish,encode_finish(finish,fy[:n*(C+H+100)]))
     else:finished=t.run('mlp_down_norm_prepared',fy[:n*(C+H+100)],[n,C],[2.,1e-6],tensor=nh['tensor'],aux=nh['aux'],input_hash=nh['input_hash'])
     assert finished.tobytes()==next_expected.tobytes()
     row.update(finish_call=t.measurements[-1],partial_down_rows=a.down_rows,follow_prepared_bitwise_equal=True,follow_history_bitwise_equal=True,follow_finished_hidden_norm_bitwise_equal=True,next_mlp_request_sha256=sha(nextreq),next_mlp_response_sha256=sha(nextres))
     finish_index=t.measurements[-1]['index'];fo=d/f'{finish_index:06d}.profile.response.bin';row['finish_profile']=t.command(dict(op='profile',input=str(d/f'{finish_index:06d}.request.bin'),output=str(fo)));assert decode(fo.read_bytes())[1].tobytes()==finished.tobytes()
     if a.finish_next_delta:
      from mlp_delta_carry import NAME as CARRY_NAME,encode_request as encode_finish
      dl=next_layer+1
      with np.load(ROOT/'artifacts/output-pairs-v2-prefix/queries/states'/f'layer-{dl:02d}.npz',allow_pickle=False)as z:dc=z['conv'].copy();dg=z['delta_log'].copy()
      dh=dict(nh,version=3,step=t.index,encoding=CARRY_NAME,op='mlp_finish_delta_log_integer',dims=[n,45,a.down_rows]);dv=t._run_encoded(dh,encode_finish(dh,fy[:n*(C+H+100)],dc,dg));row['finish_delta_call']=t.measurements[-1]
      dq=next(q for q in report['queries']if q['op']=='mlp_full_integer'and f'.layers.{dl}.'in q['tensor']);_,di=decode((source/'queries'/f"{dq['index']:06d}.request.bin").read_bytes());assert dv[:n*C].tobytes()==finished[:n*C].tobytes();assert dv[n*C:2*n*C].tobytes()==di[n*C:].tobytes()
      with np.load(ROOT/f'artifacts/output-pairs-v2-{a.source_case}/queries/states'/f'layer-{dl:02d}.npz',allow_pickle=False)as z:assert dv[2*n*C:].tobytes()==z['conv'].ravel().tobytes()
      row['finish_delta_bitwise_equal']=True
     out=d/f'{follow_index:06d}.profile.response.bin';row['follow_profile']=t.command(dict(op='profile',input=str(d/f'{follow_index:06d}.request.bin'),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==fy.tobytes()
    row.update(success=True,hidden_bitwise_equal=True,prepared_bitwise_equal=True,history_bitwise_equal=True,base_carry_bitwise_equal=True,A_carry_bitwise_equal=True)
    if layer==0 and begin==5120:
     out=d/f'{index:06d}.profile.response.bin';row['profile']=t.command(dict(op='profile',input=str(d/f'{index:06d}.request.bin'),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==y.tobytes()
    cases.append(row);print(json.dumps(row),flush=True)
 verify_module(t,wasm);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
 result=dict(scope=__doc__,wasm_sha256=wasm,source_case=a.source_case,source_hashes=hashes,cases=cases,regular_successful_diagnostic_queries=len(t.measurements),instruction_limit_diagnostic_queries=sum(c.get('instruction_limit',False)for c in cases),frame_limit_cases=sum(c.get('frame_limit',False)for c in cases),goal_50_verified=False)
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for path in paths:z.write(path,str(path.relative_to(ROOT)))
finally:t.close()
