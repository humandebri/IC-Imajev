#!/usr/bin/env python3
"""Compare real fused attention queries against saved separate kernels and native."""
import argparse,hashlib,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode,encode,atomic
from prefix_inference import verify_module

def cases(baseline):
 for label in ['prefix','617','insufficient','maximum','normal']:
  source=ROOT/f'artifacts/{baseline}-{label}';report=json.loads((source/'report.json').read_text());queries=report['queries']
  def frame(q,reply=False):return decode((source/'queries'/f'{q["index"]:06d}.{ "response" if reply else "request"}.bin').read_bytes())
  for layer in [3,31]:
   root=f'model.language_model.layers.{layer}.self_attn'
   key=next(q for q in queries if q['tensor']==root+'.k_proj.weight');kh,x=frame(key);n=kh['dims'][0]
   value=next(q for q in queries if q['tensor']==root+'.v_proj.weight');_,values=frame(value,True)
   rotation=next(q for q in queries if q['tensor']==root+'.k_norm.weight' and q['op']=='norm_rope_heads_bf16');rh,rotated=frame(rotation,True);offset=rh['dims'][3]
   expected=np.concatenate([rotated.reshape(4,n,256).transpose(1,0,2).ravel(),values])
   h=dict(kh,op='attention_kv_integer',dims=[n,offset],scalars=[],aux=[])
   yield f'{label}-layer{layer}-kv',h,x,expected
   start=next(q for q in queries if q['tensor']==root+'.q_proj.weight');qh,qinput=frame(start);qn=qh['dims'][0]
   end=next(q for q in queries if q['index']>start['index'] and q['tensor']==root+'.o_proj.weight')
   segment=[q for q in queries if start['index']<=q['index']<end['index']]
   attention=next(q for q in segment if q['op'] in ('gqa_suffix_bf16','gqa_heads_bf16'));ah,av=frame(attention)
   heads=ah['dims'][2];assert heads==16
   prefix=ah['dims'][3] if ah['op']=='gqa_suffix_bf16' else 0;total=ah['dims'][0]+prefix
   keys=av[qn*16*256:qn*16*256+4*total*256].reshape(4,total,256)
   vals=av[qn*16*256+4*total*256:].reshape(4,total,256)
   gate=[q for q in segment if q['op']=='attention_gate'];assert gate
   expected=np.concatenate([frame(q,True)[1] for q in gate]).reshape(qn,16,256)
   cap=16 if qn<=89 else 8
   for first in range(0,16,cap):
    group=first//4;groups=cap//4
    x=np.concatenate([qinput,keys[group:group+groups].ravel(),vals[group:group+groups].ravel()])
    h=dict(qh,op='attention_q_gqa_integer',dims=[qn,total,total-qn,first,cap],scalars=[],aux=[],encoding='bf16-block256-exact-v1')
    yield f'{label}-layer{layer}-q{first}',h,x,expected[:,first:first+cap].ravel()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',default='blake3-v1');ap.add_argument('--canister');ap.add_argument('--wasm');ap.add_argument('--native',required=True);ap.add_argument('--directory',required=True);ap.add_argument('--native-cache');a=ap.parse_args()
 d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());rows=[];bad=[];sha=hashlib.sha256((ROOT/a.wasm).read_bytes()).hexdigest() if a.wasm else None
 cache=json.loads((ROOT/a.native_cache/'report.json').read_text()) if a.native_cache else None
 if cache:assert cache['native_sha256']==hashlib.sha256((ROOT/a.native).read_bytes()).hexdigest() and cache['baseline']==a.baseline
 t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']) if a.canister else None
 try:
  if t:verify_module(t,sha)
  for name,h,x,expected in cases(a.baseline):
   request=d/f'{name}.request.bin';native=d/f'{name}.native.bin';reply=d/f'{name}.response.bin';atomic(request,encode(h,x))
   if cache:
    saved=next(c for c in cache['cases'] if c['name']==name)
    assert saved['request_sha256']==hashlib.sha256(request.read_bytes()).hexdigest()
    body=(ROOT/a.native_cache/f'{name}.native.bin').read_bytes()
    assert saved['native_reply_sha256']==hashlib.sha256(body).hexdigest();atomic(native,body)
   else:
    subprocess.run([str(ROOT/a.native),str(request),str(native),str(ROOT/'checkpoints/full-int8.manifest.json'),str(ROOT/'checkpoints/full-int8.pack')],check=True)
   _,got=decode(native.read_bytes());np.testing.assert_array_equal(got.view(np.uint32),expected.view(np.uint32),err_msg=name)
   row=dict(name=name,dims=h['dims'],op=h['op'],request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),native_reply_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),reference_bitwise_equal=True)
   if t:
    row['measurement']=t.command(dict(op='step',input=str(request),output=str(reply)));assert reply.read_bytes()==native.read_bytes(),name;row['wasm_native_byte_equal']=True
   rows.append(row);print(json.dumps(row),flush=True)
  if t:
   h,x=decode((d/'617-layer3-q0.request.bin').read_bytes())
   for name,modify in [('bad-position',lambda r:r.update(dims=[87,132,0,0,16])),('misaligned-head',lambda r:r.update(dims=[87,132,45,1,16])),('unknown-tensor',lambda r:r.update(tensor='bad.q_proj.weight')),('lossy',lambda r:r.update(encoding='int8-block256-v1'))]:
    r=dict(h);modify(r);request=d/f'bad-{name}.request.bin';atomic(request,encode(r,x))
    try:t.command(dict(op='step',input=str(request),output=str(d/f'bad-{name}.response.bin')))
    except RuntimeError as e:bad.append(dict(name=name,error=str(e)))
    else:raise AssertionError(name)
   verify_module(t,sha)
 finally:
  if t:t.close()
 report=dict(baseline=a.baseline,wasm_sha256=sha,native_sha256=hashlib.sha256((ROOT/a.native).read_bytes()).hexdigest(),native_cache_report_sha256=hashlib.sha256((ROOT/a.native_cache/'report.json').read_bytes()).hexdigest() if cache else None,cases=rows,negative=bad,successful_ordinary_queries=len(rows) if t else 0,rejected_ordinary_queries=len(bad),certified_module_reads=2 if t else 0,scope='Layers3/31 real saved inputs; no full-model/query-count/accuracy claim',script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
