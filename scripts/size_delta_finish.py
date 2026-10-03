#!/usr/bin/env python3
"""Size a proposed final-Delta plus output-projection query from real packets."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode,encode

def main():
    cases=[]
    for label in ['prefix','617','insufficient','maximum','normal']:
        d=ROOT/f'artifacts/activation-init-v1-{label}';report=json.loads((d/'report.json').read_text())
        def packet(q,reply=False):return (d/'queries'/f'{q["index"]:06d}.{ "response" if reply else "request"}.bin').read_bytes()
        for layer in [i for i in range(32) if i%4!=3]:
            root=f'model.language_model.layers.{layer}.linear_attn'
            reuse=next(q for q in report['queries'] if q['op']=='delta_project_reuse' and q['tensor']==root+'.in_proj_qkv.weight')
            out=next(q for q in report['queries'] if q['op']=='lora_integer' and q['tensor']==root+'.out_proj.weight')
            h,x=decode(packet(reuse));_,gated=decode(packet(out));n=h['dims'][0]
            assert h['dims'][1:3]==[16,16] and gated.size==n*4096
            first=gated.reshape(n,32,128)[:,:16].copy().ravel()
            values=np.concatenate([x,first]);header=dict(h,op='delta_project_finish')
            _,oldreply=decode(packet(reuse,True));_,projected=decode(packet(out,True))
            reply_values=np.concatenate([projected,oldreply[n*16*128:]])
            assert projected.size==n*2560
            row=dict(label=label,layer=layer,tokens=n,input_values=int(values.size),reply_values=int(reply_values.size),
                source_reuse_sha256=hashlib.sha256(packet(reuse)).hexdigest(),source_out_sha256=hashlib.sha256(packet(out)).hexdigest(),
                separate_handler_instructions=reuse['ok']['instructions']+out['ok']['instructions'])
            try:
                row.update(input_bytes=len(encode(header,values)),reply_bytes=len(encode(header,reply_values)),under_2M=True)
            except ValueError as e:row.update(under_2M=False,rejected=str(e))
            cases.append(row)
    result=dict(scope='Hypothetical query packet sizes and sum of two existing handler counters only; no implemented kernel, combined instruction/precision/query-count evidence.',cases=cases,
        script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    p=ROOT/'artifacts/column16-explicit/delta-finish-sizing.json';p.write_text(json.dumps(result,indent=2)+'\n')
    for c in cases:print(json.dumps(c),flush=True)
if __name__=='__main__':main()
