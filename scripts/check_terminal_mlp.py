#!/usr/bin/env python3
"""Verify terminal fusion against a real unfused final-layer input and outputs."""
import argparse,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module,file_hash
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);a=ap.parse_args()
 source=ROOT/'artifacts/fused-stage-617';r=json.loads((source/'report.json').read_text());q=next(q for q in r['queries'] if q['op']=='add_norm_bf16' and q['tensor'].endswith('layers.31.post_attention_layernorm.weight'))
 h,x=decode((source/'queries'/f"{q['index']:06d}.request.bin").read_bytes());n,w=h['dims'];inputs=x.reshape(2,n,w)[:,-1].ravel()
 expected=np.concatenate([np.load(source/'queries/layer-31.npy')[-1],np.load(source/'final-hidden.npy')[-1]])
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),ROOT/'artifacts/terminal-mlp-check',m['pack_hash']);t.wire_codec='bf16-exact'
 try:
  verify_module(t,file_hash(ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm'))
  y=t.run('terminal_mlp_integer',inputs,[1,2560],tensor=h['tensor'])
  np.testing.assert_array_equal(y.view(np.uint32),expected.view(np.uint32))
  print('terminal MLP hidden + normalized output bitwise equal',t.measurements[-1]['ok']['instructions'])
  (ROOT/'artifacts/terminal-mlp-check/report.json').write_text(json.dumps(t.measurements,indent=2)+'\n')
 finally:t.close()
if __name__=='__main__':main()
