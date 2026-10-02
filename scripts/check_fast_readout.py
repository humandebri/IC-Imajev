#!/usr/bin/env python3
"""Candidate widths 2..7, F32/INT8 weights independently; no new quantization."""
import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,atomic
m=json.loads((ROOT/'checkpoints/readout.manifest.json').read_text());directory=ROOT/'artifacts/fast-readout-contracts';directory.mkdir(parents=True,exist_ok=True)
t=Transport(m['model'],'http://localhost:8001/','4caro-hl777-77775-aaaba-cai',str(ROOT/'artifacts/imajev-local.pem'),directory,m['pack_hash']);hidden=json.loads((ROOT/'artifacts/reference-orders.json').read_text())['records'][0]['hidden'];rows=[]
for weight in ['readout-f32','readout-int8']:
 for count in range(2,8):
  header=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],input_hash='0'*64,step=0,op='matmul',tensor=weight,dims=[1,256,2560],scalars=[])
  path=directory/f'{weight}-{count}.bin';atomic(path,encode(header,hidden));options=[f'option-{i}' for i in range(count)];a=t.command(dict(op='decision',method='decision',input=str(path),options=options));b=t.command(dict(op='decision',method='decision_fast',input=str(path),options=options))
  for field in ['value','probabilities','unknown_probability','abstained','raw_logits','calibration_version']:assert a['ok']['decision'][field]==b['ok']['decision'][field],field
  rows.append(dict(weight=weight,count=count,identical=True,old=a,fast=b))
 path=directory/f'{weight}-2.bin'
 try:t.command(dict(op='decision',method='decision_fast',input=str(path),options=['dup','dup']))
 except RuntimeError as e:assert 'uniqueness' in str(e)
 else:raise AssertionError('duplicates accepted')
(directory/'report.json').write_text(json.dumps(dict(scope='Readout arithmetic and typed contracts, not judgment accuracy',cases=rows),indent=2)+'\n');t.close();print('12 candidate-count/precision combinations match; duplicate options rejected')
