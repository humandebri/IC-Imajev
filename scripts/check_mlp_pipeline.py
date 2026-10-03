#!/usr/bin/env python3
"""Compare two prepared-MLP queries with saved gate/down/residual-norm Wasm."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,frame_digest,encode
import struct
from mlp_codec import NAME
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest();t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec=NAME,frame_version=2);cases=[];rejected=[]
 try:
  verify_module(t,sha)
  for label in ['prefix','617','insufficient','maximum']:
   src=ROOT/f'artifacts/matrix-tail-v1-{label}';raw=(src/'report.json').read_bytes();r=json.loads(raw)
   for layer in [0,30]:
    p=f'model.language_model.layers.{layer}'
    gate=next(q for q in r['queries'] if q['op']=='mlp_add_norm_integer' and q['tensor']==p+'.mlp.gate_proj.weight')
    down=next(q for q in r['queries'] if q['op']=='lora_integer' and q['tensor']==p+'.mlp.down_proj.weight')
    norm=next(q for q in r['queries'] if q['op']=='add_norm_chain_bf16' and q['tensor']==f'model.language_model.layers.{layer+1}.input_layernorm.weight')
    request=src/'queries'/f'{gate["index"]:06d}.request.bin';h,x=decode(request.read_bytes());n=h['dims'][0]
    kw=dict(tensor=p+'.post_attention_layernorm.weight',aux=[f'model.language_model.layers.{layer+1}.input_layernorm.weight'],input_hash=h['input_hash'])
    state=t.run('mlp_prepare_down',x,[n,2560],[2.,1e-6],**kw);assert len(state)==n*11876
    got=t.run('mlp_down_norm_prepared',state,[n,2560],[2.,1e-6],**kw)
    expected=decode((src/'queries'/f'{norm["index"]:06d}.response.bin').read_bytes())[1]
    np.testing.assert_array_equal(got.view('<u4'),expected.view('<u4'),err_msg=f'{label}-{layer}')
    row=dict(label=label,layer=layer,tokens=n,bitwise_equal=True,source_report_sha256=hashlib.sha256(raw).hexdigest(),source_gate_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),queries=t.measurements[-2:],separate_handler_instructions=sum(q['ok']['instructions'] for q in [gate,down,norm]));cases.append(row);print(json.dumps(row),flush=True)
    if label=='617' and layer==0:sample=(state.copy(),n,kw.copy())
  state,n,kw=sample
  h=dict(version=2,model=m['model'],pack_hash=m['pack_hash'],input_hash=kw['input_hash'],step=0,op='mlp_down_norm_prepared',tensor=kw['tensor'],dims=[n,2560],scalars=[2.,1e-6],aux=kw['aux'],encoding=NAME)
  good=encode(h,state);hlen,=struct.unpack('<I',good[:4]);payload=good[4+hlen:-32];variants={}
  for name,index,value in [('reserved-integer',1+n*2560*2,b'\x80'),('zero-scale',1+n*2560*2+n*9216,struct.pack('<f',0.)),('nan-A',1+n*2560*2+n*9216+n*36*4,struct.pack('<f',float('nan'))),('infinite-residual',1,b'\x80\x7f')]:
   body=bytearray(payload);body[index:index+len(value)]=value;variants[name]=(h,body)
  variants['token-bound']=(dict(h,dims=[90,2560]),payload)
  variants['epsilon']=(dict(h,scalars=[2.,0.]),payload)
  variants['wrong-direction']=(dict(h,op='mlp_prepare_down'),payload)
  for name,(header,body) in variants.items():
   hb=json.dumps(header,separators=(',',':')).encode();raw=struct.pack('<I',len(hb))+hb+body;raw+=frame_digest(header,raw);request=d/f'bad-{name}.bin';request.write_bytes(raw)
   try:t.command(dict(op='step',input=str(request),output=str(d/f'bad-{name}.reply.bin')))
   except RuntimeError as e:rejected.append(dict(name=name,error=str(e)))
   else:raise AssertionError('Accepted '+name)
  verify_module(t,sha)
  (d/'report.json').write_text(json.dumps(dict(wasm_sha256=sha,cases=cases,ordinary_queries=16,rejected_ordinary_queries=len(rejected),rejected=rejected,certified_module_reads=2,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),scope='Eight real MLP boundaries against previous Wasm; full graph/query improvement not implied. Cache preparation is recorded separately.'),indent=2)+'\n')
 finally:t.close()
if __name__=='__main__':main()
