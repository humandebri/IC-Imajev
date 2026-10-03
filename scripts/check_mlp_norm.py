#!/usr/bin/env python3
"""Compare residual/norm/MLP fusion to saved real ordinary-query results."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode
from prefix_inference import verify_module


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--canister',required=True)
    ap.add_argument('--directory',required=True)
    ap.add_argument('--source',default='artifacts/packed-reduction-v1-617')
    a=ap.parse_args();dest=ROOT/a.directory;dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/a.source;r=json.loads((source/'report.json').read_text())
    manifest=ROOT/'checkpoints/full-int8.manifest.json';m=json.loads(manifest.read_text())
    headers={q['index']:decode((source/'queries'/f"{q['index']:06d}.request.bin").read_bytes())[0] for q in r['queries']}
    prefix='model.language_model.layers.0'
    before=next(q for q in r['queries'] if headers[q['index']]['tensor']==prefix+'.post_attention_layernorm.weight')
    projection=next(q for q in r['queries'] if q['op']=='mlp_gate_up_integer' and headers[q['index']]['tensor']==prefix+'.mlp.gate_proj.weight')
    after=next(q for q in r['queries'] if q['op']=='add_norm_bf16' and headers[q['index']]['tensor']=='model.language_model.layers.1.input_layernorm.weight')
    def arrays(q,suffix):return decode((source/'queries'/f"{q['index']:06d}.{suffix}.bin").read_bytes())[1]
    pair=arrays(before,'request');h=dict(headers[projection['index']]);h['op']='mlp_add_norm_integer';h['aux']=h['aux']+[prefix+'.post_attention_layernorm.weight'];h['scalars']=h['scalars']+[1e-6]
    later=arrays(after,'request');count=len(pair)//2
    chain=np.concatenate([pair,later[count:]])
    ch=dict(headers[after['index']],op='add_norm_chain_bf16')
    module_hash=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()
    t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash'])
    cases=[]
    try:
        verify_module(t,module_hash)
        for name,header,x,expected,q in [('mlp',h,pair,arrays(projection,'response'),projection),('chain',ch,chain,arrays(after,'response'),after)]:
            request=dest/f'{name}.request.bin';response=dest/f'{name}.response.bin';request.write_bytes(encode(header,x))
            got=t.command(dict(op='step',input=str(request),output=str(response)))
            _,actual=decode(response.read_bytes())
            assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),name
            case=dict(name=name,bitwise_equal=True,before_instructions=q['ok']['instructions'],**got)
            # Independent native reference for the chain; full fused MLP has a
            # scalar fixture test and the saved real two-operation Wasm oracle.
            if name=='chain':
                native=dest/'chain.native.bin';subprocess.run([str(ROOT/'target/release/primitive'),str(request),str(native),str(manifest),str(ROOT/'checkpoints/full-int8.pack')],check=True)
                _,scalar=decode(native.read_bytes());assert np.array_equal(actual.view(np.uint32),scalar.view(np.uint32));case['native_bitwise_equal']=True
            cases.append(case);print(json.dumps(case),flush=True)
        verify_module(t,module_hash)
        (dest/'report.json').write_text(json.dumps(dict(cases=cases,wasm_sha256=module_hash,source=a.source,before_norm_instructions=before['ok']['instructions'],scope='First real layer only; full graph verification required'),indent=2)+'\n')
    finally:t.close()


if __name__=='__main__':main()
