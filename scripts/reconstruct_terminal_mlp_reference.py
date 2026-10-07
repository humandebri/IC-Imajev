#!/usr/bin/env python3
"""Independent exact integer dots and ordered F32 continuation from saved MLP carries."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/terminal-mlp-reference-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bf(x):
 a=np.asarray(x,dtype='<f4');bits=a.view('<u4');return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view('<f4')
def silu(g):
 with np.errstate(over='ignore',under='ignore',invalid='raise'):
  e=bf(np.exp(np.abs(g),dtype=np.float32));y=bf(np.divide(np.float32(1),bf(np.add(np.float32(1),e)),dtype=np.float32))
  sig=np.where(g<0,y,bf(np.subtract(np.float32(1),y)))
  return bf(np.multiply(g,sig))
def ordered(x,w,initial=None):
 assert x.shape[1]==w.shape[1]
 out=np.zeros((len(x),len(w)),dtype=np.float32)if initial is None else initial.copy()
 for k in range(x.shape[1]):out=np.add(out,np.multiply(x[:,k,None],w[None,:,k]))
 return out
def quantize(x):
 a=x.reshape(len(x),-1,256);maximum=np.max(np.abs(a),axis=2)
 scale=np.where(maximum==0,np.float32(1),np.maximum(np.divide(maximum,np.float32(127)),np.array([1],dtype='<u4').view('<f4')[0])).astype('<f4')
 q=np.clip(np.rint(np.divide(a,scale[:,:,None])),-127,127).astype('<i2');return q.reshape(x.shape),scale
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args();D=BASE/('pilot'if args.pilot else'reconstruction');D.mkdir(exist_ok=False)
 extraction=json.loads((BASE/'report.json').read_text());assert extraction['complete'] and extraction['extraction_only']
 assert all(sha(ROOT/p)==h for p,h in extraction['source_hashes'].items())
 manifest=ROOT/'checkpoints/full-int8.manifest.json';m=json.loads(manifest.read_text());pack=ROOT/'checkpoints/full-int8.pack'
 tensors={t['name']:t for t in m['tensors']};weight_hashes={};weights={}
 def weight(name):
  if name not in weights:
   t=tensors[name];size=t['rows']*t['cols'];dtype='<f4'if t['dtype']=='f32'else'i1';assert t['bytes']==size*4 if dtype=='<f4'else t['bytes']==size+t['rows']*4
   with pack.open('rb')as f:f.seek(t['offset']);raw=f.read(t['bytes'])
   assert len(raw)==t['bytes'];weight_hashes[name]=hashlib.sha256(raw).hexdigest()
   w=np.frombuffer(raw,dtype=dtype,count=size).reshape(t['rows'],t['cols']);s=np.frombuffer(raw,dtype='<f4',count=t['rows'],offset=size)if dtype=='i1'else None
   if s is not None:assert np.isfinite(s).all() and np.all(s>0)
   else:assert np.isfinite(w).all()
   weights[name]=(w,s)
  return weights[name]
 def integer(q,sx,name,start=0):
  w,sw=weight(name);w=w[start:];sw=sw[start:];assert q.shape[1]==w.shape[1] and sx.shape==(len(q),q.shape[1]//256)
  out=np.zeros((len(q),len(w)),dtype=np.float32)
  for b in range(q.shape[1]//256):
   dot=q[:,b*256:(b+1)*256].astype(np.int32)@w[:,b*256:(b+1)*256].astype(np.int32).T
   out=np.add(out,np.multiply(np.multiply(dot.astype(np.float32),sx[:,b,None]),sw[None,:]))
  return out
 paid_path=ROOT/'artifacts/paid-finite-simd-v1/proof/report.json';paid=json.loads(paid_path.read_text());assert paid['complete'] and paid['baseline_restored'];latest={r['case']:r for r in paid['results']}
 cases=[r for r in extraction['cases']if r['layer']==26]+[r for r in extraction['cases']if r['layer']==30]
 if args.pilot:cases=cases[:1]
 results=[];reference_hashes={}
 try:
  for row in cases:
   started=time.monotonic();case=row['case'];layer=row['layer'];done=row['done'];root=f'model.language_model.layers.{layer}.mlp';path=ROOT/row['carry'];assert sha(path)==row['carry_sha256']
   with np.load(path,allow_pickle=False)as z:
    residual=z['residual'];q=z['q_input'];sx=z['input_scales'];gate_a=z['gate_ax'];up_a=z['up_ax'];product_q=z['product_q'];product_s=z['product_scales'];down_a=z['down_ax']
   values=[]
   for name,ax in [('gate_proj',gate_a),('up_proj',up_a)]:
    base=integer(q,sx,f'{root}.{name}.weight',done);w,_=weight(f'{root}.{name}.lora_B.weight');z=ordered(ax,w[done:]);values.append(bf(np.add(bf(base),bf(np.multiply(np.float32(2),z)))))
   product=bf(np.multiply(silu(values[0]),values[1]));assert product.shape==(row['n'],9216-done) and np.isfinite(product).all()
   w,_=weight(f'{root}.down_proj.lora_A.weight');down_a=ordered(product,w[:,done:],down_a)
   fresh_q,fresh_s=quantize(product);all_q=np.concatenate([product_q,fresh_q],axis=1);all_s=np.concatenate([product_s,fresh_s],axis=1)
   base=integer(all_q,all_s,f'{root}.down_proj.weight');w,_=weight(f'{root}.down_proj.lora_B.weight');z=ordered(down_a,w)
   y=bf(np.add(bf(base),bf(np.multiply(np.float32(2),z))));hidden=bf(np.add(residual,y));assert np.isfinite(hidden).all()
   output=D/f'{case}-layer-{layer:02d}.npy';np.save(output,hidden,allow_pickle=False)
   if layer==26:
    ref=ROOT/f'artifacts/boomdao-current-v1/{case}-r1/queries/layer-26.npy';expected=np.load(ref,allow_pickle=False)[27:];reference_hashes[str(ref.relative_to(ROOT))]=sha(ref)
    assert np.array_equal(hidden.view('<u4'),expected.view('<u4')),('historical layer26',case,int(np.count_nonzero(hidden.view('<u4')!=expected.view('<u4'))))
   else:assert len(results)>=3 and all(r['layer']==26 and r['historical_reference_bit_equal']for r in results[:3])
   current=latest[case];offset=current['quote']['prefix_tokens']-row['prefix'];actual_hash=hashlib.sha256(hidden[offset:].astype('<f4').tobytes()).hexdigest();assert actual_hash==current['debug']['hidden_hashes'][layer],('paid hidden hash',case,layer)
   result=dict(case=case,layer=layer,values=int(hidden.size),historical_reference_bit_equal=layer==26,paid_hidden_hash_equal=True,paid_module=paid['candidate'],paid_suffix_offset=offset,hidden_hash=actual_hash,output=str(output.relative_to(ROOT)),output_sha256=sha(output),seconds=time.monotonic()-started)
   results.append(result);(D/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
 except Exception as e:
  (D/'failure.json').write_text(json.dumps(dict(error=str(e),results=results,weight_hashes=weight_hashes),indent=2)+'\n');raise
 r=dict(complete=True,pilot=args.pilot,results=results,weight_hashes=weight_hashes,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),manifest,BASE/'report.json',paid_path]},reference_hashes=reference_hashes,numpy=np.__version__,scope='Independent exact I32 dots per256 and ordered F32 multiplication/addition; BF16 RNE and symmetric BF16 sigmoid/SiLU. Completes saved1280-column carry. Existing historical layer26 validates full path before missing layer30. No instruction-count claim.')
 (D/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
