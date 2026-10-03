#!/usr/bin/env python3
"""Compare Delta finish with saved separate Wasm recurrence/projection outputs."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode
from prefix_inference import verify_module

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args()
    d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);m=json.load(open(ROOT/'checkpoints/full-int8.manifest.json'));sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest()
    t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);rows=[];rejected=[]
    try:
        verify_module(t,sha)
        for label in ['prefix','617','insufficient','maximum']:
            src=ROOT/f'artifacts/column16-v1-{label}';raw=(src/'report.json').read_bytes();r=json.loads(raw)
            def packet(q,reply=False):return (src/'queries'/f'{q["index"]:06d}.{ "response" if reply else "request"}.bin').read_bytes()
            for layer in [0,30]:
                root=f'model.language_model.layers.{layer}.linear_attn'
                reuse=next(q for q in r['queries']if q['op']=='delta_project_reuse' and q['tensor']==root+'.in_proj_qkv.weight')
                out=next(q for q in r['queries']if q['op']=='lora_integer' and q['tensor']==root+'.out_proj.weight')
                h,x=decode(packet(reuse));_,gated=decode(packet(out));n=h['dims'][0]
                first=gated.reshape(n,32,128)[:,:16].copy().ravel();values=np.concatenate([x,first]);header=dict(h,op='delta_project_finish')
                _,oldreply=decode(packet(reuse,True));_,projected=decode(packet(out,True));expected=np.concatenate([projected,oldreply[n*16*128:]])
                name=f'{label}-{layer}';request=d/f'{name}.request.bin';reply=d/f'{name}.reply.bin';request.write_bytes(encode(header,values))
                measured=t.command(dict(op='step',input=str(request),output=str(reply)));got=decode(reply.read_bytes())[1]
                np.testing.assert_array_equal(got.view(np.uint32),expected.view(np.uint32),err_msg=name)
                rows.append(dict(name=name,bitwise_equal=True,source_report_sha256=hashlib.sha256(raw).hexdigest(),request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),reply_sha256=hashlib.sha256(reply.read_bytes()).hexdigest(),measurement=measured,separate_handler_instructions=reuse['ok']['instructions']+out['ok']['instructions']))
                print(json.dumps(rows[-1]),flush=True)
                if label=='617' and layer==0:sample=(header,values.copy(),len(x))
        h,values,reuse_len=sample;variants={}
        for name,dims in [('too-long',[91,16,16,0]),('wrong-first',[87,16,0,0]),('wrong-heads',[87,14,16,0]),('wrong-keep',[87,16,16,2])]:variants[name]=(dict(h,dims=dims),values)
        bad=values.copy();bad[-1]=1.0000001;variants['non-bf16-first']=(h,bad)
        for name,value in [('fractional',.5),('reserved',-128.)]:
            bad=values.copy();bad[87*2560-1]=value;variants[name]=(h,bad)
        for name,(header,values)in variants.items():
            request=d/f'bad-{name}.bin';request.write_bytes(encode(header,values))
            try:t.command(dict(op='step',input=str(request),output=str(d/f'bad-{name}.reply.bin')))
            except RuntimeError as e:rejected.append(dict(name=name,error=str(e)))
            else:raise AssertionError('Accepted '+name)
        verify_module(t,sha)
        (d/'report.json').write_text(json.dumps(dict(wasm_sha256=sha,cases=rows,rejected=rejected,ordinary_queries=len(rows),rejected_ordinary_queries=len(rejected),certified_module_reads=2,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),scope='8 real recurrence/projection boundaries; full graph accuracy/query counts not implied'),indent=2)+'\n')
    finally:t.close()
if __name__=='__main__':main()
