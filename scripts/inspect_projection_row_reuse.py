#!/usr/bin/env python3
"""Inspect exact row identity opportunities; no execution savings are claimed."""
from pathlib import Path
import hashlib,json,struct,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import frame_digest
from mlp_stream_codec import decode_payload,NAME
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/projection-row-reuse-v1';d.mkdir(exist_ok=False)
 paid_path=ROOT/'artifacts/paid-finite-simd-v1/proof/report.json';paid=json.loads(paid_path.read_text());assert paid['complete']
 paths=[Path(__file__),paid_path,ROOT/'client/transport.py',ROOT/'client/mlp_stream_codec.py',ROOT/'client/projection_codec.py',ROOT/'artifacts/update-finite-simd-v1/build/runtime/delta_mlp_start.rs'];cases=[]
 for item in paid['results']:
  case=item['case'];prefix=item['quote']['prefix_tokens'];tokens=item['request']['request']['token_ids'][prefix:];n=len(tokens)
  base=ROOT/f'artifacts/boomdao-current-v1/{case}-r1';report_path=base/'report.json';report=json.loads(report_path.read_text());paths.append(report_path);layers={}
  for query in report['queries']:
   layer=int(query['tensor'].split('.')[3])
   if layer in layers:continue
   p=base/'queries'/f"{query['index']:06d}.request.bin";raw=p.read_bytes();size=struct.unpack_from('<I',raw)[0];h=json.loads(raw[4:4+size])
   if h['encoding']!=NAME or h['op']!='mlp_stream_complete':continue
   assert frame_digest(h,raw[:-32])==raw[-32:] and h['model']==paid['results'][0]['request']['request']['model']
   rows=h['dims'][0];assert rows==len(item['request']['request']['token_ids'])-27
   v=decode_payload(h,raw[4+size:-32]);q=v[rows*2560:2*rows*2560].reshape(rows,2560).astype('<i2');scales=v[2*rows*2560:2*rows*2560+rows*10].reshape(rows,10)
   offset=prefix-27;keys=[q[t].tobytes()+scales[t].astype('<f4').tobytes()for t in range(offset,rows)]
   assert len(keys)==n;layers[layer]=dict(layer=layer,rows=n,unique_quantized_rows=len(set(keys)),duplicate_rows=n-len(set(keys)),source=str(p.relative_to(ROOT)),sha256=sha(p));paths.append(p)
  assert len(layers)>0
  cases.append(dict(case=case,suffix_tokens=n,unique_token_ids=len(set(tokens)),repeated_token_ids=n-len(set(tokens)),repeated_token_percent=100*(1-len(set(tokens))/n),mlp_input_rows=list(layers.values())))
 r=dict(complete=True,inspection_only=True,cases=cases,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in paths},scope='Token ID repeats imply identical embedding and norm0 rows before position/context processing. Saved MLP carry q+scale rows compared by exact byte equality after current prefix trimming. No projection sharing implementation or instruction savings claimed.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-inspection.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in list(dict.fromkeys(paths))+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps([dict(case=c['case'],suffix_tokens=c['suffix_tokens'],repeated_token_ids=c['repeated_token_ids'],mlp_layers_inspected=len(c['mlp_input_rows']),duplicate_mlp_rows=sum(x['duplicate_rows']for x in c['mlp_input_rows']))for c in cases],indent=2))
if __name__=='__main__':main()
