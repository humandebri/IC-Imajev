#!/usr/bin/env python3
"""Compare complete MLP against previous deployed pipeline/final-layer outputs."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--main-baseline');a=ap.parse_args();d=ROOT/a.directory
 m=json.load(open(ROOT/'checkpoints/full-int8.manifest.json'));sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest();t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='bf16-block256-exact-v1',frame_version=2);cases=[]
 try:
  verify_module(t,sha)
  for label,layers in [('prefix',[0,30,31]),('insufficient',[0,30])]+([('617',list(range(31)))] if a.main_baseline else []):
   src=ROOT/f'artifacts/{a.main_baseline if label=="617" else "attention-full-v1"}-{label}';raw=(src/'report.json').read_bytes();r=json.loads(raw)
   def frame(q,reply=False):return decode((src/'queries'/f'{q["index"]:06d}.{ "response" if reply else "request"}.bin').read_bytes())
   for layer in layers:
    root=f'model.language_model.layers.{layer}';name=root+'.post_attention_layernorm.weight';next_norm=f'model.language_model.layers.{layer+1}.input_layernorm.weight' if layer<31 else 'model.language_model.norm.weight'
    if layer<31:
     start=next(q for q in r['queries'] if q['op']=='mlp_prepare_down' and q['tensor']==name);end=next(q for q in r['queries'] if q['op']=='mlp_down_norm_prepared' and q['tensor']==name);separate=[start,end]
    else:
     start=next(q for q in r['queries'] if q['op']=='mlp_add_norm_integer' and q['tensor']==root+'.mlp.gate_proj.weight');down=next(q for q in r['queries'] if q['op']=='lora_integer' and q['tensor']==root+'.mlp.down_proj.weight');end=next(q for q in r['queries'] if q['op']=='add_norm_chain_bf16' and q['tensor']==next_norm);separate=[start,down,end]
    h,x=frame(start);n=h['dims'][0];expected=frame(end,True)[1]
    got=t.run('mlp_full_integer',x,[n,2560],[2.,1e-6],tensor=name,aux=[next_norm],input_hash=h['input_hash']);np.testing.assert_array_equal(got.view('<u4'),expected.view('<u4'),err_msg=f'{label}-{layer}')
    row=dict(label=label,layer=layer,tokens=n,bitwise_equal=True,query=t.measurements[-1],separate_instructions=sum(q['ok']['instructions'] for q in separate),source_report_sha256=hashlib.sha256(raw).hexdigest());cases.append(row);print(json.dumps(row),flush=True)
  rejected=[]
  for label,dims,tensor,aux,scalars in [('token-bound',[88,2560],name,[next_norm],[2.,1e-6]),('layer',[n,2560],'model.language_model.layers.32.post_attention_layernorm.weight',[next_norm],[2.,1e-6]),('epsilon',[n,2560],name,[next_norm],[2.,0.]),('next-norm',[n,2560],name,['model.language_model.norm.weight'],[2.,1e-6])]:
   try:t.run('mlp_full_integer',x,dims,scalars,tensor=tensor,aux=aux,input_hash=h['input_hash'])
   except RuntimeError as e:rejected.append(dict(label=label,error=str(e)))
   else:raise AssertionError('Accepted '+label)
  verify_module(t,sha);(d/'report.json').write_text(json.dumps(dict(wasm_sha256=sha,cases=cases,ordinary_queries=len(cases),rejected_ordinary_queries=len(rejected),rejected=rejected,certified_module_reads=2,scope='Complete MLP boundaries and optional all 31 main-question pipeline layers against previous Wasm; full graph evidence separate'),indent=2)+'\n')
 finally:t.close()
if __name__=='__main__':main()
