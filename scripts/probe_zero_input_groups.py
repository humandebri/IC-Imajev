#!/usr/bin/env python3
"""Measure exactly-zero quantized K4 groups; never feeds host data to inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode

source=ROOT/'artifacts/mlp-reuse-v1-617'
dest=ROOT/'artifacts/prefix_codec/zero-group-probe'
dest.mkdir(parents=True,exist_ok=True)
helper=ROOT/'artifacts/bounded_i16/native/release/quantized_input'
report=json.loads((source/'report.json').read_bytes())
cases=[]
for query in report['queries']:
    if not query.get('tensor','').endswith('.mlp.down_proj.weight'):
        continue
    request=source/'queries'/f'{query["index"]:06d}.request.bin'
    header,values=decode(request.read_bytes())
    n,rows,cols,start=header['dims']
    assert header['op']=='lora_integer' and values.size==n*cols and cols%256==0
    path=dest/f'{query["index"]:06d}'
    path.with_suffix('.input.bin').write_bytes(values.astype('<f4').tobytes())
    subprocess.run([str(helper),str(n),str(cols),str(path.with_suffix('.input.bin')),
                    str(path.with_suffix('.q.bin'))],check=True)
    q=np.frombuffer(path.with_suffix('.q.bin').read_bytes(),np.int8).reshape(n,cols)
    groups=np.all(q.reshape(n,cols//4,4)==0,axis=2)
    blocks=np.all(q.reshape(n,cols//256,256)==0,axis=2)
    cases.append(dict(tensor=header['tensor'],tokens=n,cols=cols,
        source_request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
        zero_fraction=float(np.mean(q==0)),zero_k4_fraction=float(np.mean(groups)),
        zero_k256_fraction=float(np.mean(blocks))))
result=dict(scope='Saved main suffix down-projection inputs. Host-only exact zero counting, not a Wasm speedup or accuracy measurement.',
    source_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
    probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    quantizer_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),cases=cases)
(dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),zero_k4_range=[min(c['zero_k4_fraction'] for c in cases),max(c['zero_k4_fraction'] for c in cases)])))
