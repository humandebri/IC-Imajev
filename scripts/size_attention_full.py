#!/usr/bin/env python3
"""Size proposed complete attention queries from validated packets; no inference."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode,encode

def main():
    cases=[]
    for label in ['prefix','617','insufficient','maximum','normal']:
        d=ROOT/f'artifacts/delta-finish-v2-{label}'
        report=json.loads((d/'report.json').read_text())
        def packet(q,reply=False):return (d/'queries'/f'{q["index"]:06d}.{ "response" if reply else "request"}.bin').read_bytes()
        for layer in range(3,32,4):
            root=f'model.language_model.layers.{layer}.self_attn'
            kv=next(q for q in report['queries'] if q['op']=='attention_kv_integer' and q['tensor']==root+'.k_proj.weight')
            qs=[q for q in report['queries'] if q['op']=='attention_q_gqa_integer' and q['tensor']==root+'.q_proj.weight']
            op=next(q for q in report['queries'] if q['op']=='lora_integer' and q['tensor']==root+'.o_proj.weight')
            h,x=decode(packet(kv));n,prefix=h['dims'];qh,qx=decode(packet(qs[0]));qn,total,offset,first,heads=qh['dims']
            assert first==0 and total==prefix+n and len(x)==n*2560
            groups=heads//4;part=groups*total*256
            keys=qx[qn*2560:qn*2560+part].reshape(groups,total,256)[:,:prefix]
            values=qx[qn*2560+part:].reshape(groups,total,256)[:,:prefix]
            # Long inputs split Q across two GQA groups; append their prefix
            # groups without changing group order or any numeric values.
            if len(qs)>1:
                _,tail=decode(packet(qs[1]));tg=qs[1]
                th,_=decode(packet(tg));assert th['dims']==[qn,total,offset,8,8]
                keys=np.concatenate([keys,tail[qn*2560:qn*2560+part].reshape(groups,total,256)[:,:prefix]])
                values=np.concatenate([values,tail[qn*2560+part:].reshape(groups,total,256)[:,:prefix]])
            assert keys.shape==(4,prefix,256)
            inp=np.concatenate([x,keys.ravel(),values.ravel()])
            _,kr=decode(packet(kv,True));_,projected=decode(packet(op,True))
            assert len(kr)==n*2048 and len(projected)==qn*2560
            header=dict(h,op='attention_full_integer',dims=[n,prefix,int(qn==1)],tensor=root+'.q_proj.weight')
            out=np.concatenate([projected,kr])
            separate=[kv,*qs,op]
            count=sum(q['ok']['instructions'] for q in separate)
            row=dict(label=label,layer=layer,tokens=n,q_tokens=qn,input_values=int(inp.size),reply_values=int(out.size),input_bytes=len(encode(header,inp)),reply_bytes=len(encode(header,out)),separate_queries=len(separate),separate_handler_instructions=count,source_packets={str(q['index']):hashlib.sha256(packet(q)).hexdigest() for q in separate})
            row['frame_fit']=row['input_bytes']<=2_000_000 and row['reply_bytes']<=2_000_000
            row['separate_under_5B']=count<5_000_000_000
            cases.append(row)
    result=dict(scope='Host packet sizing and sum of existing separate handler counters only; no fused Wasm implementation, query limit proof, precision result or measured reduction.',cases=cases)
    dest=ROOT/'artifacts/column16-token48/attention-full-sizing.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    for label in ['prefix','617','insufficient','maximum','normal']:
        cs=[c for c in cases if c['label']==label]
        print(json.dumps(dict(label=label,frame_fits=sum(c['frame_fit'] for c in cs),separate_under_5B=sum(c['separate_under_5B'] for c in cs),maximum_separate_instructions=max(c['separate_handler_instructions'] for c in cs),maximum_input_bytes=max(c['input_bytes'] for c in cs),maximum_reply_bytes=max(c['reply_bytes'] for c in cs))))
if __name__=='__main__':main()
