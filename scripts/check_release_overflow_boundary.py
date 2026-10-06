#!/usr/bin/env python3
"""Record rejection semantics for an overflowing normalization shape in a separate query."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode
from prefix_inference import verify_module

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--module',required=True);ap.add_argument('--label',required=True);a=ap.parse_args()
    d=ROOT/'artifacts/release-overflow-v1/full-proof-v1/boundary'/a.label;d.mkdir(parents=True,exist_ok=False)
    m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());h=dict(version=3,model=m['model'],pack_hash=m['pack_hash'],input_hash='0'*64,step=0,op='rms_scaled',tensor='',dims=[262144,16384],scalars=[1e-6,1.],encoding='bf16-block256-exact-v1')
    request=d/'request.bin';request.write_bytes(encode(h,np.array([],dtype='<f4')))
    t=Transport(m['model'],'http://localhost:8001/','6eydd-o3777-77775-aaama-cai',str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],bridge_binary=str(ROOT/'artifacts/query-packing-v3/build/imajev-client'))
    try:
        verify_module(t,a.module)
        try:reply=t.command(dict(op='step',input=str(request),output=str(d/'response.bin')));result=dict(accepted=True,reply=reply)
        except RuntimeError as error:result=dict(accepted=False,error=str(error))
        verify_module(t,a.module)
    finally:t.close()
    result.update(module=a.module,request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),separate_ordinary_queries=1,scope='Invalid shape: element count equals 2^32 in Wasm32. Independent of normal inference replay.')
    (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
