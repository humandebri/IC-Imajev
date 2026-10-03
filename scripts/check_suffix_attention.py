#!/usr/bin/env python3
"""Actual Wasm regression: causal key offset must not offset the output slice."""
import argparse,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module,file_hash
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);a=ap.parse_args()
 source=ROOT/'artifacts/fused-stage-617';report=json.loads((source/'report.json').read_text());q=next(q for q in report['queries'] if q['op']=='gqa_heads_bf16')
 path=source/'queries'/f"{q['index']:06d}";h,x=decode(path.with_suffix('.request.bin').read_bytes());_,expected=decode(path.with_suffix('.response.bin').read_bytes());n,w,heads=h['dims'];query=x[:heads*n*w].reshape(heads,n,w);kv=x[heads*n*w:]
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/suffix-check',m['pack_hash']);t.wire_codec='bf16-exact'
 try:
  verify_module(t,file_hash(ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm'))
  for offset in [0,45,n-1]:
   y=t.run('gqa_suffix_bf16',np.concatenate([query[:,offset:].ravel(),kv]),[n-offset,w,heads,offset])
   np.testing.assert_array_equal(y.view(np.uint32),expected.reshape(heads,n,w)[:,offset:].copy().ravel().view(np.uint32))
   print('suffix',offset,'bitwise equal',t.measurements[-1]['ok']['instructions'],flush=True)
  (ROOT/'artifacts/suffix-check/report.json').write_text(json.dumps(t.measurements,indent=2)+'\n')
 finally:t.close()
if __name__=='__main__':main()
