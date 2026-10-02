#!/usr/bin/env python3
"""Same real projection inputs/weights, baseline and candidate Wasm; no Laya calls."""
import argparse,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);args=ap.parse_args();dest=ROOT/'artifacts/linear-tiling'/args.phase;dest.mkdir(parents=True,exist_ok=True)
r=json.loads((ROOT/'artifacts/activation-int8-fused-617/first-report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());source=ROOT/'artifacts/activation-int8-fused-617/queries';seen=set();chosen=[]
for q in r['queries']:
 if q['op'] not in ('lora_project','linear_bf16'):continue
 h,_=decode((source/f"{q['index']:06d}.request.bin").read_bytes());key=(q['op'],*h['dims'][:3])
 if key in seen:continue
 seen.add(key);chosen.append(q)
 if len(chosen)==8:break
t=Transport(m['model'],'http://localhost:8001/','46el7-ql777-77775-aaada-cai',str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash']);records=[]
try:
 for q in chosen:
  i=q['index'];request=source/f'{i:06d}.request.bin';response=dest/f'{i:06d}.response.bin';start=time.perf_counter();got=t.command(dict(op='step',input=str(request),output=str(response)));wall=time.perf_counter()-start
  if 'ok' not in got:raise RuntimeError(got)
  h,a=decode(response.read_bytes());_,b=decode((source/f'{i:06d}.response.bin').read_bytes());equal=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)));assert equal,(i,float(abs(a-b).max()))
  rec=dict(index=i,op=q['op'],dims=h['dims'],tensor=q['tensor'],bitwise_equal=equal,**got['ok'],wall_seconds=wall,request_sha256=hashlib.sha256(request.read_bytes()).hexdigest());records.append(rec);print(rec,flush=True)
 wasm=(ROOT/'artifacts/linear-tiling/baseline.wasm') if args.phase=='baseline' else (ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm')
 summary=dict(model=m['model'],pack_hash=m['pack_hash'],wasm_sha256=hashlib.sha256(wasm.read_bytes()).hexdigest(),phase=args.phase,query_cache_controlled=False,cases=records);(dest/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
finally:t.close()
