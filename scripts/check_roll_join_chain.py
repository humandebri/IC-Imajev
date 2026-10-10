#!/usr/bin/env python3
"""Measure five consecutive client-held joins from MLP completion through two next layers."""
import argparse,hashlib,io,json,pathlib,shutil,sys,zipfile
import numpy as np
if not __debug__:raise RuntimeError('Proof checker requires assertions; do not use Python -O')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_bytes,read_report,selections,validate_join_settings
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
from mlp_delta_carry import HUFFMAN_NAME,encode_request as finish
from attention_mlp_stream_codec import NAME as ATT,C,KV,encode_request as attention
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request
from mlp_stream_codec import NAME as STREAM
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--cases',default='617,insufficient,maximum');ap.add_argument('--layers',default='1');ap.add_argument('--heads',default='24,26');ap.add_argument('--down-rows',type=int,default=768);ap.add_argument('--entry-start',action='store_true',help='Include preceding Delta/MLP start, using canister embedding for layer0');ap.add_argument('--entry-heads',type=int,help='Obtain the partial-down carry from the preceding real MLP/Delta join');ap.add_argument('--entry-front',type=int,default=4096);ap.add_argument('--front',type=int,default=1792);ap.add_argument('--attention-front',type=int,default=5376);ap.add_argument('--compress-base',action='store_true');ap.add_argument('--compact-attention',action='store_true');ap.add_argument('--compress-residual',action='store_true');ap.add_argument('--residual-raw-threshold',type=float,default=0.);ap.add_argument('--residual-dictionary',action='store_true');a=ap.parse_args();validate_join_settings(a);d=ROOT/a.directory
if (d/'report.json').exists():raise ValueError('Use a new evidence directory')
d.mkdir(parents=True,exist_ok=True)
labels=selections(a.cases,('617','insufficient','maximum'),'cases');layers=list(map(int,selections(a.layers,tuple(map(str,(1,5,9,13,17,21,25))),'layers')));heads=list(map(int,selections(a.heads,tuple(map(str,range(2,32,2))),'heads')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());module=sha(ROOT/a.wasm)
base=ROOT/'artifacts/prefix_codec/full-attention-q4-proof-v1';paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'canisters/inference/Cargo.toml',ROOT/'scripts/build_full_prefix_candidate.py',pathlib.Path(__file__),ROOT/'scripts/proof_inputs.py'];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};references={};rows=[]
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='mlp-down-state-exact-v1',frame_version=3)
def reference(label,layer):
 root=base/label;report=read_report(root/'report.json',ROOT,references);q=next(q for q in report['queries']if q['op']=='mlp_full_integer'and f'.layers.{layer}.'in q['tensor']);req=root/'queries'/f"{q['index']:06d}.request.bin";res=root/'queries'/f"{q['index']:06d}.response.bin";h,x=decode(read_bytes(req,ROOT,references));_,both=decode(read_bytes(res,ROOT,references));return h,x,both

def state(label,layer,prefix=False):
 path=(base/'prefix'if prefix else base/label)/'queries/states'/f'layer-{layer:02d}.npz';raw=read_bytes(path,ROOT,references)
 with np.load(io.BytesIO(raw),allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def header(h,op,encoding,dims):return dict(h,version=3,step=t.index,op=op,encoding=encoding,dims=dims)
def run(h,payload,scope):
 index=t.index
 try:result=t._run_encoded(h,payload)
 except RuntimeError as error:
  if not is_instruction_limit(error):raise
  p=d/f'failed-{scope}.request.bin';shutil.copyfile(d/f'{index:06d}.request.bin',p);rows.append(dict(scope=scope,success=False,stage=h['op'],error=str(error),failed_request=p.name,failed_request_sha256=sha(p)));return None
 metric=t.measurements[-1]
 (d/f'{index:06d}.metric.json').write_text(json.dumps(metric,indent=2)+'\n')
 return result,metric,d/f'{index:06d}.request.bin'
def profile(record,expected):
 _,_,req=record;out=req.with_name(req.stem+'.profile.response.bin');r=t.command(dict(diagnostics=True,op='profile',input=str(req),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==expected.tobytes();req.with_name(req.stem+'.profile.json').write_text(json.dumps(r,indent=2)+'\n');return r
try:
 verify_module(t,module)
 for label in labels:
  for layer in layers:
   assert layer in [1,5,9,13,17,21,25]
   h1,x1,both1=reference(label,layer);h2,x2,both2=reference(label,layer+1);h3,x3,both3=reference(label,layer+2);h4,x4,both4=reference(label,layer+3);n=h1['dims'][0]
   t.wire_codec='mlp-down-state-exact-v1';prepared=t.run('mlp_prepare_partial_down',x1,[n,C,a.down_rows],[2.,1e-6],tensor=h1['tensor'],aux=h1['aux'],input_hash=h1['input_hash'])
   entry_calls=[];entry_profiles=[]
   if a.entry_heads is not None:
    eh=a.entry_heads;eb=a.entry_front
    h0,x0,both0=reference(label,layer-1);t.wire_codec=STREAM
    prev=t.run('mlp_stream_prepare',x0,[n,0,eb],[2.,1e-6],tensor=h0['tensor'],aux=h0['aux'],input_hash=h0['input_hash'])
    if a.entry_start:
     from delta_mlp_start_codec import NAME as START,CONV,encode_ids,encode_request as start_packet
     start_prefix=state(label,layer-1,True)
     ehead=header(h0,'delta_mlp_stream_start_ids'if layer==1 else'delta_mlp_stream_prepare',START,[n,eb,45])
     if layer==1:
      serving=read_report(ROOT/'artifacts/reference-serving.json',ROOT,references)
      if serving['model_lock_sha256']!=m['model']:raise ValueError('start reference model')
      ids=serving['records'][{'617':0,'insufficient':19,'maximum':11}[label]]['token_ids'][45:]
      if len(ids)!=n:raise ValueError('start token length')
      packet=encode_ids(ehead,ids,start_prefix['conv'],start_prefix['delta_log'])
     else:
      _,_,prior=reference(label,layer-2)
      packet=start_packet(ehead,prior[:n*C],prior[n*C:],start_prefix['conv'],start_prefix['delta_log'])
     entry=run(ehead,packet,f'{label}-l{layer}-entry-start')
     if entry is None:continue
     ey=entry[0];assert ey[:-CONV].tobytes()==prev.tobytes();assert ey[-CONV:].tobytes()==state(label,layer-1)['conv'].ravel().tobytes()
     entry_profiles.append(profile(entry,ey));entry_calls.append(entry[1]);prev=ey[:-CONV]
    ep=state(label,layer,True);ec=ep['conv'];el=ep['delta_log'];ef=np.concatenate([np.arange(eh//2*128),np.arange(2048,2048+eh//2*128),np.arange(4096,4096+eh*128)]);er=np.concatenate([np.arange(eh//2*128,2048),np.arange(2048+eh//2*128,4096),np.arange(4096+eh*128,8192)])
    def entry_log(first):
     sl=slice(None,eh//2*128)if first else slice(eh//2*128,None);vl=slice(None,eh*128)if first else slice(eh*128,None);gl=slice(None,eh)if first else slice(eh,None)
     return np.concatenate([el[:45*2048].reshape(45,2048)[:,sl].ravel(),el[45*2048:45*6144].reshape(45,4096)[:,vl].ravel(),el[45*6144:].reshape(45,32)[:,gl].ravel()])
    ehead=header(h0,'mlp_complete_delta_partial',PAIR,[n,eb,9216-eb,eh,45]);entry=run(ehead,pair(ehead,prev,ec[:,ef],entry_log(True)),f'{label}-l{layer}-entry-first')
    if entry is None:continue
    ey=entry[0];assert ey[:n*C].tobytes()==both0[:n*C].tobytes();assert ey[-3*eh*256:].tobytes()==state(label,layer)['conv'][:,ef].ravel().tobytes();entry_profiles.append(profile(entry,ey));entry_calls.append(entry[1])
    ehead=header(h0,'delta_partial_mlp_prepare_down',PAIR,[n,eb,9216-eb,eh,45,a.down_rows]);entry=run(ehead,encode_continue_request(ehead,ey,ec[:,er],entry_log(False),compress_base=True),f'{label}-l{layer}-entry-down')
    if entry is None:continue
    ey=entry[0];assert ey[:n*(C+9216+100)].tobytes()==prepared.tobytes();assert ey[-3*(32-eh)*256:].tobytes()==state(label,layer)['conv'][:,er].ravel().tobytes();entry_profiles.append(profile(entry,ey));entry_calls.append(entry[1])
    prepared=ey[:n*(C+9216+100)]

   prefix2=state(label,layer+1,True);h=header(h1,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,45,a.down_rows,a.front]);first=run(h,finish(h,prepared,prefix2['conv'],prefix2['delta_log']),f'{label}-l{layer}-front')
   if first is None:continue
   y=first[0];assert y[:n*C].tobytes()==both1[:n*C].tobytes();assert y[-24576:].tobytes()==state(label,layer+1)['conv'].ravel().tobytes();carry=y[n*C:-24576]
   t.wire_codec=STREAM;expected=t.run('mlp_stream_prepare',x2,[n,0,a.front],[2.,1e-6],tensor=h2['tensor'],aux=h2['aux'],input_hash=h2['input_hash']);assert carry.tobytes()==expected.tobytes();fp=profile(first,y)
   z=state(label,layer+2,True);prefix=np.concatenate([z['keys'].transpose(1,0,2).ravel(),z['values'].transpose(1,0,2).ravel()]);h=header(h2,'mlp_complete_attention_kv_q4',ATT,[n,a.front,9216-a.front,45]);second=run(h,attention(h,carry,prefix),f'{label}-l{layer}-q4')
   if second is None:continue
   y=second[0];assert y[:2*n*C].tobytes()==both2.tobytes();sp=profile(second,y)
   h=header(h3,'attention_finish_mlp_front_q4_compact'if a.compact_attention else'attention_finish_mlp_front_q4',ATT,[n,a.attention_front,45]);third=run(h,attention(h,y,prefix),f'{label}-l{layer}-q12')
   if third is None:continue
   carry=third[0]if a.compact_attention else third[0][:-n*KV];t.wire_codec=STREAM;expected=t.run('mlp_stream_prepare',x3,[n,0,a.attention_front],[2.,1e-6],tensor=h3['tensor'],aux=h3['aux'],input_hash=h3['input_hash']);assert carry.tobytes()==expected.tobytes();tp=profile(third,third[0]);prefix4=state(label,layer+3,True);cv=prefix4['conv'];log=prefix4['delta_log'];final=state(label,layer+3)['conv']
   for k in heads:
    first_cols=np.concatenate([np.arange(k//2*128),np.arange(2048,2048+k//2*128),np.arange(4096,4096+k*128)]);rest_cols=np.concatenate([np.arange(k//2*128,2048),np.arange(2048+k//2*128,4096),np.arange(4096+k*128,8192)])
    def sliced(start,end):return np.concatenate([log[:45*2048].reshape(45,2048)[:,start//2*128:end//2*128].ravel(),log[45*2048:45*6144].reshape(45,4096)[:,start*128:end*128].ravel(),log[45*6144:].reshape(45,32)[:,start:end].ravel()])
    h=header(h3,'mlp_complete_delta_partial',PAIR,[n,a.attention_front,9216-a.attention_front,k,45]);scope=f'{label}-l{layer}-heads{k}-pair'
    try:packet=pair(h,carry,cv[:,first_cols],sliced(0,k),compress_residual=a.compress_residual,residual_raw_threshold=a.residual_raw_threshold,residual_dictionary=a.residual_dictionary)
    except ValueError as error:
     if 'frame bounds'not in str(error):raise
     (d/f'failed-{scope}.header.json').write_text(json.dumps(h,indent=2)+'\n');row=dict(scope=scope,label=label,layer=layer,heads=k,success=False,stage='pair_frame',frame_limit=True,error=str(error),carry_response=third[2].with_name(third[2].name.replace('.request.bin','.response.bin')).name,header_file=f'failed-{scope}.header.json');rows.append(row);print(json.dumps(row),flush=True);continue
    fourth=run(h,packet,scope)
    if fourth is None:continue
    y=fourth[0];assert y[:n*C].tobytes()==both3[:n*C].tobytes();assert y[-3*k*256:].tobytes()==final[:,first_cols].ravel().tobytes();qp=profile(fourth,y)
    h=header(h3,'delta_partial_mlp_full',PAIR,[n,a.attention_front,9216-a.attention_front,k,45]);fifth=run(h,encode_continue_request(h,y,cv[:,rest_cols],sliced(k,32),compress_base=a.compress_base),f'{label}-l{layer}-heads{k}-full')
    if fifth is None:continue
    y=fifth[0];assert y[:2*n*C].tobytes()==both4.tobytes();assert y[2*n*C:].tobytes()==final[:,rest_cols].ravel().tobytes();lastp=profile(fifth,y)
    calls=[r[1]for r in [first,second,third,fourth,fifth]];row=dict(label=label,layer=layer,tokens=n,heads=k,front=a.front,down_rows=a.down_rows,attention_front=a.attention_front,compress_base=a.compress_base,compact_attention=a.compact_attention,compress_residual=a.compress_residual,residual_raw_threshold=a.residual_raw_threshold,residual_dictionary=a.residual_dictionary,entry_start=a.entry_start,entry_heads=a.entry_heads,entry_front=a.entry_front,entry_calls=entry_calls,entry_profiles=entry_profiles,entry_bitwise_equal=True if entry_calls else None,success=True,exported_hidden_norm_bitwise_equal=True,prepared_stream_bitwise_equal=True,conv_bitwise_equal=True,calls=calls,profiles=[fp,sp,tp,qp,lastp],five_query_instructions=sum(c['ok']['instructions']for c in calls),five_query_candid_bytes=sum(c['ok']['request_bytes']+c['ok']['reply_bytes']for c in calls));rows.append(row);print(json.dumps(row),flush=True)
 verify_module(t,module)
 if sha(ROOT/a.wasm)!=module:raise ValueError('Wasm file changed during run')
 if hashes!={str(p.relative_to(ROOT)):sha(p)for p in paths}:raise ValueError('Source changed during run')
 if not all(sha(ROOT/p)==v for p,v in references.items()):raise ValueError('Reference changed during run')
 (d/'report.json').write_text(json.dumps(dict(settings=vars(a),module_sha256=module,source_hashes=hashes,reference_hashes=references,cases=rows,measurements=t.measurements,whole_inference_reduction_verified=False,requested_chains=len(labels)*len(layers)*len(heads),completed_chains=sum(row.get('success')is True for row in rows),failed_attempts=sum(row.get('success')is False for row in rows)),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
