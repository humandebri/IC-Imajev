#!/usr/bin/env python3
"""Bound a proposed exact INT8/F32/BF16 prepared-MLP frame; no inference."""
import hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode

def main():
    out=[]
    for label in ['prefix','617','insufficient','maximum']:
        d=ROOT/f'artifacts/matrix-tail-v1-{label}';r=json.loads((d/'report.json').read_text())
        for layer in range(31):
            root=f'model.language_model.layers.{layer}'
            gate=next(q for q in r['queries'] if q['op']=='mlp_add_norm_integer' and q['tensor']==root+'.mlp.gate_proj.weight')
            down=next(q for q in r['queries'] if q['op']=='lora_integer' and q['tensor']==root+'.mlp.down_proj.weight')
            norm=next(q for q in r['queries'] if q['op']=='add_norm_chain_bf16' and q['tensor']==f'model.language_model.layers.{layer+1}.input_layernorm.weight')
            def packet(q):return (d/'queries'/f'{q["index"]:06d}.request.bin').read_bytes()
            g,x=decode(packet(gate));dh,product=decode(packet(down));nh,residuals=decode(packet(norm));n=g['dims'][0]
            assert g['dims']==[n,9216,2560,0] and dh['dims']==[n,2560,9216,0] and nh['dims']==[n,2560]
            assert len(x)==n*5120 and len(product)==n*9216 and len(residuals)==n*7680
            # Independent typed segments, not a raised bound for ordinary
            # flat-float packets. Integer bytes are already-quantized product;
            # scales/A products keep original F32; added residual keeps BF16.
            qbytes=n*9216;scale_bytes=n*36*4;ax_bytes=n*64*4;residual_bytes=n*2560*2
            prepared_bound=16424+4*4+qbytes+scale_bytes+ax_bytes+residual_bytes
            row=dict(label=label,layer=layer,tokens=n,prepared_int8_values=qbytes,remaining_f32_values=n*100,residual_bf16_values=n*2560,flat_logical_values=n*(9216+100+2560),prepared_frame_bound=prepared_bound,first_request_bound=16424+n*5120*2,last_reply_bound=16424+n*5120*2,separate_handler_instructions=sum(q['ok']['instructions'] for q in [gate,down,norm]),gate_handler_instructions=gate['ok']['instructions'],down_norm_handler_instructions=down['ok']['instructions']+norm['ok']['instructions'],source_packets={str(q['index']):hashlib.sha256(packet(q)).hexdigest() for q in [gate,down,norm]})
            assert qbytes<=900000 and n*100<=900000 and n*2560<=900000
            assert max(prepared_bound,row['first_request_bound'],row['last_reply_bound'])<=2_000_000
            out.append(row)
    result=dict(scope='Conservative typed-packet byte bounds from real shapes; no host inference, new quantization, fused kernel, instruction-limit proof or measured query reduction. Existing flat codec900K remains unchanged; new private typed segments would require their own checked decoder.',cases=out)
    (ROOT/'artifacts/matrix-tail/prepared-mlp-sizing.json').write_text(json.dumps(result,indent=2)+'\n')
    for label in ['prefix','617','insufficient','maximum']:
        cases=[c for c in out if c['label']==label]
        print(json.dumps(dict(label=label,cases=len(cases),max_prepared_bytes=max(c['prepared_frame_bound'] for c in cases),max_gate_instructions=max(c['gate_handler_instructions'] for c in cases),max_down_norm_instructions=max(c['down_norm_handler_instructions'] for c in cases),max_whole_mlp_instructions=max(c['separate_handler_instructions'] for c in cases))))
if __name__=='__main__':main()
