#!/usr/bin/env python3
"""Measure five consecutive client-held joins from MLP completion through two next layers."""
import argparse,hashlib,json,pathlib,shutil,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,is_instruction_limit
from prefix_inference import verify_module
from mlp_delta_carry import HUFFMAN_NAME,encode_request as finish
from attention_mlp_stream_codec import NAME as ATT,C,KV,encode_request as attention
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request
from mlp_stream_codec import NAME as STREAM
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument('--cases',default='617,insufficient,maximum');ap.add_argument('--layers',default='1');ap.add_argument('--heads',default='24,26');ap.add_argument('--down-rows',type=int,default=768);ap.add_argument('--front',type=int,default=1792);ap.add_argument('--attention-front',type=int,default=5376);ap.add_argument('--compress-base',action='store_true');ap.add_argument('--compact-attention',action='store_true');ap.add_argument('--compress-residual',action='store_true');ap.add_argument('--residual-raw-threshold',type=float,default=0.);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);assert not(d/'report.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());module=sha(ROOT/a.wasm)
base=ROOT/'artifacts/prefix_codec/full-attention-q4-proof-v1';paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};references={};rows=[]
t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='mlp-down-state-exact-v1',frame_version=3)
def reference(label,layer):
 root=base/label;report=json.loads((root/'report.json').read_text());q=next(q for q in report['queries']if q['op']=='mlp_full_integer'and f'.layers.{layer}.'in q['tensor']);req=root/'queries'/f"{q['index']:06d}.request.bin";res=root/'queries'/f"{q['index']:06d}.response.bin";references[str(req.relative_to(ROOT))]=sha(req);references[str(res.relative_to(ROOT))]=sha(res);h,x=decode(req.read_bytes());_,both=decode(res.read_bytes());return h,x,both

def state(label,layer,prefix=False):
 path=(base/'prefix'if prefix else base/label)/'queries/states'/f'layer-{layer:02d}.npz';references[str(path.relative_to(ROOT))]=sha(path)
 with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
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
 _,_,req=record;out=req.with_name(req.stem+'.profile.response.bin');r=t.command(dict(op='profile',input=str(req),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==expected.tobytes();req.with_name(req.stem+'.profile.json').write_text(json.dumps(r,indent=2)+'\n');return r
try:
 verify_module(t,module)
 for label in a.cases.split(','):
  for layer in map(int,a.layers.split(',')):
   assert layer in [1,5,9,13,17,21,25]
   h1,x1,both1=reference(label,layer);h2,x2,both2=reference(label,layer+1);h3,x3,both3=reference(label,layer+2);h4,x4,both4=reference(label,layer+3);n=h1['dims'][0]
   t.wire_codec='mlp-down-state-exact-v1';prepared=t.run('mlp_prepare_partial_down',x1,[n,C,a.down_rows],[2.,1e-6],tensor=h1['tensor'],aux=h1['aux'],input_hash=h1['input_hash'])
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
   for k in map(int,a.heads.split(',')):
    first_cols=np.concatenate([np.arange(k//2*128),np.arange(2048,2048+k//2*128),np.arange(4096,4096+k*128)]);rest_cols=np.concatenate([np.arange(k//2*128,2048),np.arange(2048+k//2*128,4096),np.arange(4096+k*128,8192)])
    def sliced(start,end):return np.concatenate([log[:45*2048].reshape(45,2048)[:,start//2*128:end//2*128].ravel(),log[45*2048:45*6144].reshape(45,4096)[:,start*128:end*128].ravel(),log[45*6144:].reshape(45,32)[:,start:end].ravel()])
    h=header(h3,'mlp_complete_delta_partial',PAIR,[n,a.attention_front,9216-a.attention_front,k,45]);scope=f'{label}-l{layer}-heads{k}-pair'
    try:packet=pair(h,carry,cv[:,first_cols],sliced(0,k),compress_residual=a.compress_residual,residual_raw_threshold=a.residual_raw_threshold)
    except ValueError as error:
     if 'frame bounds'not in str(error):raise
     (d/f'failed-{scope}.header.json').write_text(json.dumps(h,indent=2)+'\n');row=dict(scope=scope,label=label,layer=layer,heads=k,success=False,stage='pair_frame',frame_limit=True,error=str(error),carry_response=third[2].with_name(third[2].name.replace('.request.bin','.response.bin')).name,header_file=f'failed-{scope}.header.json');rows.append(row);print(json.dumps(row),flush=True);continue
    fourth=run(h,packet,scope)
    if fourth is None:continue
    y=fourth[0];assert y[:n*C].tobytes()==both3[:n*C].tobytes();assert y[-3*k*256:].tobytes()==final[:,first_cols].ravel().tobytes();qp=profile(fourth,y)
    h=header(h3,'delta_partial_mlp_full',PAIR,[n,a.attention_front,9216-a.attention_front,k,45]);fifth=run(h,encode_continue_request(h,y,cv[:,rest_cols],sliced(k,32),compress_base=a.compress_base),f'{label}-l{layer}-heads{k}-full')
    if fifth is None:continue
    y=fifth[0];assert y[:2*n*C].tobytes()==both4.tobytes();assert y[2*n*C:].tobytes()==final[:,rest_cols].ravel().tobytes();lastp=profile(fifth,y)
    calls=[r[1]for r in [first,second,third,fourth,fifth]];row=dict(label=label,layer=layer,tokens=n,heads=k,front=a.front,down_rows=a.down_rows,attention_front=a.attention_front,compress_base=a.compress_base,compact_attention=a.compact_attention,compress_residual=a.compress_residual,residual_raw_threshold=a.residual_raw_threshold,success=True,exported_hidden_norm_bitwise_equal=True,prepared_stream_bitwise_equal=True,conv_bitwise_equal=True,calls=calls,profiles=[fp,sp,tp,qp,lastp],five_query_instructions=sum(c['ok']['instructions']for c in calls),five_query_candid_bytes=sum(c['ok']['request_bytes']+c['ok']['reply_bytes']for c in calls));rows.append(row);print(json.dumps(row),flush=True)
 verify_module(t,module);assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths};assert all(sha(ROOT/p)==v for p,v in references.items())
 (d/'report.json').write_text(json.dumps(dict(module_sha256=module,source_hashes=hashes,reference_hashes=references,cases=rows,measurements=t.measurements,whole_inference_reduction_verified=False),indent=2)+'\n')
 with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
finally:t.close()
