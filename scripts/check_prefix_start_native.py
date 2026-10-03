#!/usr/bin/env python3
"""Compare fused token-ID native output to separate operators in the same build.

Saved Wasm boundary differences are reported separately. Full Wasm inference
parity is established by check_full_prefix_hybrid, not this native check.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import zipfile
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode,encode
from prefix_start import NAME,encode_request
from prefix_hybrid import encode_request as encode_delta,NAME as DELTA_NAME


def main():
    directory=ROOT/'artifacts/prefix_codec/native-prefix-start-check'
    directory.mkdir(parents=True,exist_ok=True)
    helper=ROOT/'artifacts/prefix_codec/native-target/release/primitive'
    source_root=ROOT/'artifacts/prefix_codec/full-owned-state-proof'
    sha=lambda b:hashlib.sha256(b).hexdigest()
    paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'client').glob('*.py'))+[pathlib.Path(__file__)]
    hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
    cases=[]
    for label in ['617','insufficient','maximum']:
        source=source_root/label;report_bytes=(source/'report.json').read_bytes();report=json.loads(report_bytes)
        embed=next(q for q in report['queries'] if q['op']=='embed')
        delta=next(q for q in report['queries'] if q['op']=='delta_full_hybrid_integer' and '.layers.0.' in q['tensor'])
        request=source/'queries'/f'{embed["index"]:06d}.request.bin'
        embed_header,ids=decode(request.read_bytes());n=len(ids)
        with np.load(source_root/'prefix/queries/states/layer-00.npz',allow_pickle=False) as state:
            conv=state['conv'].copy()
        packet_path=source_root/'packets/layer-00.npf1';packet=packet_path.read_bytes()
        header=dict(embed_header,op='prefix_start_integer',encoding=NAME,dims=[n,45])
        fused=encode_request(header,np.concatenate([ids,conv.ravel()]),packet)
        path=directory/f'{label}.request.bin';path.write_bytes(fused)
        output=directory/f'{label}.response.bin'
        subprocess.run([str(helper),str(path),str(output),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True,cwd=ROOT)
        returned,values=decode(output.read_bytes())
        assert returned==dict(header,step=header['step']+1)
        embed_response=(source/'queries'/f'{embed["index"]:06d}.response.bin').read_bytes()
        delta_response=(source/'queries'/f'{delta["index"]:06d}.response.bin').read_bytes()
        def run_operator(kind,frame):
            source_path=directory/f'{label}.{kind}.request.bin';source_path.write_bytes(frame)
            target=directory/f'{label}.{kind}.response.bin'
            subprocess.run([str(helper),str(source_path),str(target),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True,cwd=ROOT)
            return decode(target.read_bytes())[1]
        native_embed=run_operator('embed',request.read_bytes())
        norm_request=source/'queries'/f'{embed["index"]+1:06d}.request.bin'
        norm_header,_=decode(norm_request.read_bytes())
        assert norm_header['op']=='rms_bf16'
        native_norm=run_operator('norm',encode(norm_header,native_embed))
        delta_request=(source/'queries'/f'{delta["index"]:06d}.request.bin').read_bytes()
        delta_header=json.loads(delta_request[4:4+int.from_bytes(delta_request[:4],'little')])
        assert delta_header['encoding']==DELTA_NAME
        native_delta=run_operator('delta',encode_delta(delta_header,np.concatenate([native_norm,conv.ravel()]),packet))
        expected=np.concatenate([native_embed,native_delta])
        assert values.tobytes()==expected.tobytes(),label
        saved_wasm=np.concatenate([decode(embed_response)[1],decode(delta_response)[1]])
        cases.append(dict(label=label,tokens=n,bitwise_equal_to_same_build_separate_operators=True,
                          saved_wasm_boundary_differing_values=int(np.count_nonzero(values.view('<u4')!=saved_wasm.view('<u4'))),
                          saved_wasm_boundary_max_abs_error=float(np.max(np.abs(values-saved_wasm))),
                          request_sha256=sha(fused),source_report_sha256=sha(report_bytes),packet_sha256=sha(packet),embedding_response_sha256=sha(embed_response),delta_response_sha256=sha(delta_response),output_sha256=sha(output.read_bytes()),values_sha256=sha(values.tobytes())))
    assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
    result=dict(scope=__doc__,helper_sha256=sha(helper.read_bytes()),source_hashes=hashes,cases=cases)
    (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(directory/'validated-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps(cases))


if __name__=='__main__':main()
