#!/usr/bin/env python3
"""Inspect exact I32 dot reuse by K256 lanes, without merging F32 scale arithmetic."""
from pathlib import Path
import hashlib,json,struct,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import frame_digest
from mlp_stream_codec import decode_payload,NAME
D=ROOT/'artifacts/projection-block-reuse-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False);up=ROOT/'artifacts/projection-row-reuse-v1/report.json';previous=json.loads(up.read_text());assert previous['complete']
 assert all(sha(ROOT/p)==h for p,h in previous['source_hashes'].items())
 paths=[Path(__file__),up,ROOT/'client/transport.py',ROOT/'client/mlp_stream_codec.py',ROOT/'client/projection_codec.py'];out=[]
 for case in previous['cases']:
  paid=json.loads((ROOT/'artifacts/paid-finite-simd-v1/proof/report.json').read_text());item=next(x for x in paid['results']if x['case']==case['case']);offset=item['quote']['prefix_tokens']-27
  for layer in case['mlp_input_rows']:
   p=ROOT/layer['source'];assert sha(p)==layer['sha256'];raw=p.read_bytes();size=struct.unpack_from('<I',raw)[0];h=json.loads(raw[4:4+size]);assert h['encoding']==NAME and frame_digest(h,raw[:-32])==raw[-32:]
   rows=h['dims'][0];v=decode_payload(h,raw[4+size:-32]);q=v[rows*2560:2*rows*2560].reshape(rows,2560).astype('<i2')[offset:];sc=v[2*rows*2560:2*rows*2560+rows*10].reshape(rows,10).astype('<f4')[offset:]
   assert len(q)==case['suffix_tokens'];blocks=[]
   for b in range(10):
    keys=[row.tobytes()for row in q[:,b*256:(b+1)*256]];with_sc=[key+sc[t,b].tobytes()for t,key in enumerate(keys)]
    blocks.append(dict(block=b,unique_lanes=len(set(keys)),unique_lanes_and_scale=len(set(with_sc))))
   whole=len(set(row.tobytes()for row in q));saved=sum(len(q)-b['unique_lanes']for b in blocks)
   out.append(dict(case=case['case'],layer=layer['layer'],tokens=len(q),whole_q_unique=whole,blocks=blocks,saved_dot_rows=saved,total_dot_rows=len(q)*10,source=layer['source'],sha256=layer['sha256']));paths.append(p)
 aggregate={c:dict(saved=sum(x['saved_dot_rows']for x in out if x['case']==c),total=sum(x['total_dot_rows']for x in out if x['case']==c))for c in ['617','620','653']}
 r=dict(complete=True,inspection_only=True,results=out,aggregate=aggregate,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in paths},scope='Saved exact MLP input carries only. Equal q lanes imply equal integer dots; every row must retain original F32 scales, ordered K256 accumulation, LoRA and activation. No performance claim.')
 (D/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-inspection.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(paths+[D/'report.json']):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(aggregate))
if __name__=='__main__':main()
