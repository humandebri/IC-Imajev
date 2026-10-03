#!/usr/bin/env python3
"""Compare complete attention ordinary queries with saved deployed outputs."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory
 m=json.load(open(ROOT/'checkpoints/full-int8.manifest.json'));sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest();t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='bf16-block256-exact-v1',frame_version=2);cases=[]
 try:
  verify_module(t,sha)
  for label in ['prefix','617','insufficient','maximum','normal']:
   src=ROOT/f'artifacts/dot-scale-v1-{label}';raw=(src/'report.json').read_bytes();r=json.loads(raw)
   def frame(q,reply=False):return decode((src/'queries'/f'{q["index"]:06d}.{ "response" if reply else "request"}.bin').read_bytes())
   for layer in ([31] if label=='normal' else [3,31]):
    root=f'model.language_model.layers.{layer}.self_attn'
    kv=next(q for q in r['queries'] if q['op']=='attention_kv_integer' and q['tensor']==root+'.k_proj.weight')
    qs=[q for q in r['queries'] if q['op']=='attention_q_gqa_integer' and q['tensor']==root+'.q_proj.weight'];assert len(qs)==1
    op=next(q for q in r['queries'] if q['op']=='lora_integer' and q['tensor']==root+'.o_proj.weight')
    h,x=frame(kv);n,prefix=h['dims'];qh,qx=frame(qs[0]);qn,total,offset,first,heads=qh['dims'];assert first==0 and heads==16 and total==n+prefix
    part=4*total*256;keys=qx[qn*2560:qn*2560+part].reshape(4,total,256)[:,:prefix];values=qx[qn*2560+part:].reshape(4,total,256)[:,:prefix]
    payload=np.concatenate([x,keys.ravel(),values.ravel()]);expected=np.concatenate([frame(op,True)[1],frame(kv,True)[1]])
    got=t.run('attention_full_integer',payload,[n,prefix,int(qn==1)],tensor=root+'.q_proj.weight',input_hash=h['input_hash']);np.testing.assert_array_equal(got.view('<u4'),expected.view('<u4'),err_msg=f'{label}-{layer}')
    row=dict(label=label,layer=layer,tokens=n,q_tokens=qn,bitwise_equal=True,query=t.measurements[-1],separate_instructions=sum(q['ok']['instructions'] for q in [kv,*qs,op]),source_report_sha256=hashlib.sha256(raw).hexdigest());cases.append(row);print(json.dumps(row),flush=True)
  rejected=[]
  for name,dims,tensor,scalars in [('nonterminal-long',[132,0,0],root+'.q_proj.weight',[]),('terminal-flag',[132,0,2],root+'.q_proj.weight',[]),('prefix-length',[132,1,1],root+'.q_proj.weight',[]),('layer','same','model.language_model.layers.30.self_attn.q_proj.weight',[]),('scalars','same',root+'.q_proj.weight',[2.])]:
   if dims=='same':dims=[132,0,1]
   try:t.run('attention_full_integer',payload,dims,scalars,tensor=tensor,input_hash=h['input_hash'])
   except RuntimeError as e:rejected.append(dict(name=name,error=str(e)))
   else:raise AssertionError('Accepted invalid '+name)
  verify_module(t,sha);(d/'report.json').write_text(json.dumps(dict(wasm_sha256=sha,cases=cases,ordinary_queries=len(cases),rejected_ordinary_queries=len(rejected),rejected=rejected,certified_module_reads=2,scope='Nine complete attention boundaries against previous Wasm; all-layer evidence recorded separately'),indent=2)+'\n')
 finally:t.close()
if __name__=='__main__':main()
