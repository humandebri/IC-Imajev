#!/usr/bin/env python3
"""Compare one full-width MLP query against every saved output slice of a layer."""
import argparse, hashlib, json, pathlib, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,encode
from prefix_inference import verify_module
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--source',default='artifacts/kernel-unroll-v1-maximum');ap.add_argument('--output',default='artifacts/wide-mlp89');a=ap.parse_args()
 source=ROOT/a.source;dest=ROOT/a.output;dest.mkdir(parents=True,exist_ok=True)
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());r=json.loads((source/'report.json').read_text())
 first=next(q for q in r['queries'] if q['op']=='mlp_gate_up_integer');parts=[q for q in r['queries'] if q['op']=='mlp_gate_up_integer' and q['tensor']==first['tensor']]
 values=[];headers=[];before=0;x=None
 for q in parts:
  p=source/'queries'/f"{q['index']:06d}";h,input_values=decode(p.with_suffix('.request.bin').read_bytes());_,v=decode(p.with_suffix('.response.bin').read_bytes());n,width,cols,start=h['dims'];assert n==89 and cols==2560
  if x is None:x=input_values
  else:assert np.array_equal(x.view(np.uint32),input_values.view(np.uint32))
  values.append((start,v.reshape(n,width)));headers.append(h);before+=q['ok']['instructions']
 values.sort();expected=np.concatenate([v for _,v in values],axis=1);assert expected.shape==(89,9216)
 h=headers[0];h['dims']=[89,9216,2560,0];request=dest/'request.bin';request.write_bytes(encode(h,x));out=dest/'response.bin'
 t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash']);wasm=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()
 try:
  verify_module(t,wasm);result=t.command(dict(op='step',input=str(request),output=str(out)));_,v=decode(out.read_bytes());assert np.array_equal(v.view(np.uint32),expected.ravel().view(np.uint32));verify_module(t,wasm)
  report=dict(bitwise_equal=True,tokens=89,before_queries=len(parts),after_queries=1,before_instructions=before,after_instructions=result['ok']['instructions'],wasm_sha256=wasm);(dest/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
 finally:t.close()
if __name__=='__main__':main()
