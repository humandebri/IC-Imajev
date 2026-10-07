#!/usr/bin/env python3
"""Extract saved exact MLP carry to reconstruct the missing historical layer30 hidden."""
from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import frame_digest
from mlp_stream_codec import decode_payload,NAME,length
D=ROOT/'artifacts/terminal-mlp-reference-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());rows=[];sources=[Path(__file__),ROOT/'client/transport.py',ROOT/'client/mlp_stream_codec.py',ROOT/'checkpoints/full-int8.manifest.json']
 for case in ['617','620','653']:
  base=ROOT/f'artifacts/boomdao-current-v1/{case}-r1';report=json.loads((base/'report.json').read_text());sources.append(base/'report.json')
  for layer in [26,30]:
   op='mlp_stream_complete_terminal'if layer==30 else'mlp_stream_complete_attention_full'
   q=[x for x in report['queries']if x.get('op')==op and x['tensor'].split('.')[3]==str(layer)];assert len(q)==1
   source=base/'queries'/f"{q[0]['index']:06d}.request.bin";sources.append(source);raw=source.read_bytes();size=struct.unpack_from('<I',raw)[0];h=json.loads(raw[4:4+size]);assert frame_digest(h,raw[:-32])==raw[-32:]
   assert h['model']==m['model'] and h['pack_hash']==m['pack_hash'] and h['op']==op
   n,p,done=h['dims'];assert p==27 and done%256==0 and 0<done<9216
   payload=raw[4+size:-32];assert payload[0]==3;span=struct.unpack_from('<I',payload,1)[0];assert len(payload)==5+span+2*p*2048
   inner=dict(h,encoding=NAME,op='mlp_stream_complete',dims=[n,done,9216-done]);v=decode_payload(inner,payload[5:5+span]);assert len(v)==length(n,done)
   cursor=0
   def take(count,shape):
    nonlocal cursor
    result=v[cursor:cursor+count].reshape(shape);cursor+=count;return result
   residual=take(n*2560,(n,2560));q_input=take(n*2560,(n,2560));input_scales=take(n*10,(n,10));gate_ax=take(n*64,(n,64));up_ax=take(n*64,(n,64));product_q=take(n*done,(n,done));product_scales=take(n*(done//256),(n,done//256));down_ax=take(n*64,(n,64));assert cursor==len(v)
   path=D/f'{case}-layer-{layer:02d}.npz';np.savez(path,residual=residual,q_input=q_input.astype('<i2'),input_scales=input_scales,gate_ax=gate_ax,up_ax=up_ax,product_q=product_q.astype('<i2'),product_scales=product_scales,down_ax=down_ax)
   if layer==26:
    reference=base/'queries'/f'layer-{layer:02d}.npy';assert np.load(reference,allow_pickle=False).shape==(n+p,2560);sources.append(reference)
   rows.append(dict(case=case,layer=layer,n=n,prefix=p,done=done,header=h,source=str(source.relative_to(ROOT)),source_sha256=sha(source),carry=str(path.relative_to(ROOT)),carry_sha256=sha(path),reference_reconstructed=False))
 r=dict(complete=True,extraction_only=True,cases=rows,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},scope='Saved prefix27 MLP carries for layers26/30. Exact residual, quantized lanes/scales, LoRA-A products, product prefix and unrounded partial down-A. Layer26 supplies an existing independent hidden reference to validate reconstruction. No hidden30 equality claim yet.')
 (D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps([dict(case=x['case'],layer=x['layer'],n=x['n'],done=x['done'])for x in rows]))
if __name__=='__main__':main()
