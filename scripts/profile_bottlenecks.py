#!/usr/bin/env python3
"""Profile representative real queries; compare response bits to saved full inference.
Spans are inclusive. They are not added to produce a total.
"""
import argparse,collections,hashlib,json,pathlib,sys,time
import numpy as np
import struct
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--canister',default='46el7-ql777-77775-aaada-cai');p.add_argument('--normal-only',action='store_true');p.add_argument('--integer-only',action='store_true');p.add_argument('--source-directory',default='artifacts/integer-full-617');args=p.parse_args()
r=json.loads((ROOT/args.source_directory/'first-report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());source=ROOT/args.source_directory/'queries';dest=ROOT/'artifacts/bottleneck'/args.phase;dest.mkdir(parents=True,exist_ok=True)
groups=collections.defaultdict(lambda:dict(count=0,instructions=0,bytes=0,seconds=0));unique={}
for q in r['queries']:
 g=groups[q['op']];g['count']+=1;g['instructions']+=q['ok']['instructions'];g['bytes']+=q['ok']['request_bytes']+q['ok']['reply_bytes'];g['seconds']+=q['wall_seconds']
 with (source/f"{q['index']:06d}.request.bin").open('rb') as frame:
  size,=struct.unpack('<I',frame.read(4));h=json.loads(frame.read(size))
 key=(q['op'],tuple(h['dims'][:3]))
 if key not in unique:unique[key]=q
# All distinct full-model integer projection shapes, and one of each other op.
chosen=[];seen=set()
for key,q in unique.items():
 if 'integer' in q['op'] or q['op'] not in seen:chosen.append(q);seen.add(q['op'])
if args.integer_only:chosen=[q for q in chosen if 'integer' in q['op']]
t=Transport(m['model'],'http://localhost:8001/',args.canister,str(ROOT/'artifacts/imajev-local.pem'),dest,m['pack_hash']);cases=[]
try:
 for q in chosen:
  i=q['index'];request=source/f'{i:06d}.request.bin';normal=dest/f'{i:06d}.normal.bin';profile=dest/f'{i:06d}.profile.bin'
  h,_=decode(request.read_bytes());a=t.command(dict(op='step',input=str(request),output=str(normal)));b=t.command(dict(diagnostics=True,op='profile',input=str(request),output=str(profile))) if not args.normal_only else a
  if 'ok' not in a or 'ok' not in b:raise RuntimeError((a,b))
  _,x=decode(normal.read_bytes());_,y=decode((source/f'{i:06d}.response.bin').read_bytes());_,z=decode(profile.read_bytes() if not args.normal_only else normal.read_bytes());assert np.array_equal(x.view(np.uint32),y.view(np.uint32)) and np.array_equal(x.view(np.uint32),z.view(np.uint32)),i
  c=dict(index=i,op=q['op'],tensor=q['tensor'],dims=h['dims'],bitwise_equal=True,baseline_instructions=q['ok']['instructions'],normal=a['ok'],profile=b['ok'],profile_overhead=b['ok']['instructions']-a['ok']['instructions'],profiling_enabled=not args.normal_only,normal_wall_seconds=a.get('wall_seconds'),profile_wall_seconds=b.get('wall_seconds'));cases.append(c);print(q['op'],h['dims'],a['ok']['instructions'],b['ok'].get('spans',[]),flush=True)
 out=dict(phase=args.phase,canister=args.canister,url='http://localhost:8001/',wasm_sha256=hashlib.sha256((ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm').read_bytes()).hexdigest(),full_baseline_total=r['total_instructions'],op_totals=groups,cases=cases,scope='Real saved inputs, instruction counters, inclusive spans, profiling overhead reported separately; no hardware wall-time attribution')
 (dest/'report.json').write_text(json.dumps(out,indent=2)+'\n')
finally:t.close()
