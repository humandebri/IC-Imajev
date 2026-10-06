#!/usr/bin/env python3
"""Compare real cold-model capture replies, including exact reusable q/scale/A state."""
import argparse,hashlib,json,pathlib,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for p in ['canister','wasm','directory']:ap.add_argument('--'+p,required=True)
    a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source=ROOT/'artifacts/prefix_codec/full-f32-k-continue-proof/normal'
    report=json.loads((source/'report.json').read_bytes())
    paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))
    paths += [pathlib.Path(__file__),ROOT/'MODEL_LOCK.json',ROOT/'Cargo.lock',ROOT/'checkpoints/full-int8.manifest.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};wasm=sha(ROOT/a.wasm)
    m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes())
    t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],wire_codec='projection-block256-exact-v1',frame_version=3)
    cases=[]
    try:
        verify_module(t,wasm)
        for q in report['queries']:
            if q['op']!='mlp_gate_up_capture':continue
            request=source/'queries'/f"{q['index']:06d}.request.bin"
            response=source/'queries'/f"{q['index']:06d}.response.bin"
            h,x=decode(request.read_bytes());_,expected=decode(response.read_bytes())
            actual=t.run(h['op'],x,h['dims'],h['scalars'],tensor=h['tensor'],input_hash=h['input_hash'],aux=h['aux'])
            assert actual.dtype==expected.dtype and actual.shape==expected.shape and actual.tobytes()==expected.tobytes()
            metric=t.measurements[-1]['ok']
            assert metric['instructions']<q['ok']['instructions']
            row=dict(tensor=h['tensor'],dims=h['dims'],bitwise_equal=True,source_request_sha256=sha(request),source_response_sha256=sha(response),before_instructions=q['ok']['instructions'],after_instructions=metric['instructions'],saved_instructions=q['ok']['instructions']-metric['instructions'],candid_bytes=metric['request_bytes']+metric['reply_bytes'])
            cases.append(row);print(json.dumps(row),flush=True)
        assert len(cases)==31
        verify_module(t,wasm);assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in paths}
        result=dict(scope=__doc__,wasm_sha256=wasm,model=m['model'],pack_hash=m['pack_hash'],cases=cases,source_hashes=hashes,ordinary_diagnostic_queries=len(t.measurements),saved_instructions=sum(c['saved_instructions'] for c in cases),goal_50_verified=False)
        (d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(p,str(p.relative_to(ROOT)))
    finally:t.close()

if __name__=='__main__':main()
