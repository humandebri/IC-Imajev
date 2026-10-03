#!/usr/bin/env python3
"""Validate compact wire operations against saved real outputs and long-input heads."""
import argparse,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module,file_hash

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);args=ap.parse_args()
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
 t=Transport(m['model'],'http://localhost:8001/',args.canister,str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/compact-head-check',m['pack_hash'],'bf16-exact');cases=[]
 def equal(a,b):assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
 try:
  verify_module(t,file_hash(ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm'))
  for source in ['artifacts/layout-full-617','artifacts/prefix-hit-617']:
   directory=ROOT/source;report=json.loads((directory/'first-report.json').read_text())
   queries=[q for q in report['queries'] if q['op']=='delta_heads_bf16']
   for q in [queries[0],queries[-1]]:
    path=directory/'queries'/f"{q['index']:06d}"
    h,x=decode(path.with_suffix('.request.bin').read_bytes());_,old=decode(path.with_suffix('.response.bin').read_bytes())
    n,k,v,heads=h['dims'];a=t.run('delta_terminal_bf16',x,h['dims']);equal(a,old[:heads*n*v]);cases.append(dict(source=source,op='terminal',index=q['index'],bitwise_equal=True))
   queries=[q for q in report['queries'] if q['op']=='attention_heads_bf16'][:2];inputs=[];outputs=[]
   for q in queries:
    path=directory/'queries'/f"{q['index']:06d}";h,x=decode(path.with_suffix('.request.bin').read_bytes());_,y=decode(path.with_suffix('.response.bin').read_bytes());n,w,heads=h['dims'];inputs.append(x.reshape(heads,3,n,w));outputs.append(y)
   a=np.concatenate(inputs);packed=np.concatenate([a[:,0].ravel(),a[::4,1].ravel(),a[::4,2].ravel()]);got=t.run('gqa_heads_bf16',packed,[n,w,len(a)]);equal(got,np.concatenate(outputs));cases.append(dict(source=source,op='gqa',heads=len(a),bitwise_equal=True))
  for n,heads in [(147,12),(512,2)]:
   stride=n*256;x=(np.arange((heads+2*((heads+3)//4))*stride,dtype=np.float32)%7)/8
   got=t.run('gqa_heads_bf16',x,[n,256,heads]).reshape(heads,-1)
   for head in range(heads):
    inputs=np.concatenate([x[head*stride:(head+1)*stride],x[(heads+head//4)*stride:(heads+head//4+1)*stride],x[(heads+(heads+3)//4+head//4)*stride:(heads+(heads+3)//4+head//4+1)*stride]])
    equal(got[head],t.run('attention_bf16',inputs,[n,256]))
   cases.append(dict(op='gqa_boundary',tokens=n,heads=heads,bitwise_equal=True))
  (ROOT/'docs/compact-head-check.json').write_text(json.dumps(dict(cases=cases,queries=t.measurements),indent=2)+'\n');print(cases)
 finally:t.close()
if __name__=='__main__':main()
