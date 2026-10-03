#!/usr/bin/env python3
"""Measure the deployed lossless block codec with fixed real numerical inputs."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode
from prefix_inference import verify_module


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--source',default='artifacts/mlp-norm-v1-617');a=ap.parse_args()
    dest=ROOT/a.directory;dest.mkdir(parents=True,exist_ok=True);source=ROOT/a.source
    r=json.loads((source/'report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    sha=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest()
    t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash'])
    seen=set();cases=[]
    try:
        verify_module(t,sha)
        for q in r['queries']:
            if q['op'] in seen:continue
            seen.add(q['op']);p=source/'queries'/f"{q['index']:06d}";h,x=decode(p.with_suffix('.request.bin').read_bytes())
            results={};outputs={}
            for codec in ['bf16-exact','bf16-block256-exact-v1']:
                header=dict(h,encoding=codec);request=dest/f"{q['index']}-{codec}.request.bin";response=dest/f"{q['index']}-{codec}.response.bin"
                request.write_bytes(encode(header,x));result=t.command(dict(op='step',input=str(request),output=str(response)))
                rh,values=decode(response.read_bytes());_,expected=decode(p.with_suffix('.response.bin').read_bytes())
                assert np.array_equal(values.view(np.uint32),expected.view(np.uint32)),(q['op'],codec)
                results[codec]=result;outputs[codec]=values
                # Cross-language canonical payload check, using the response's
                # actual header; JSON numeric spelling can differ independently.
                encoded=encode(rh,values);raw=response.read_bytes();n=int.from_bytes(raw[:4],'little');pn=int.from_bytes(encoded[:4],'little')
                assert raw[4+n:-32]==encoded[4+pn:-32]
            case=dict(op=q['op'],dims=h['dims'],bitwise_equal=True,canonical_payload_equal=True,results=results)
            cases.append(case);print(json.dumps(case),flush=True)
        verify_module(t,sha)
        (dest/'report.json').write_text(json.dumps(dict(cases=cases,wasm_sha256=sha,scope='Representative real ops on same deployed module; not full-graph improvement'),indent=2)+'\n')
    finally:t.close()


if __name__=='__main__':main()
