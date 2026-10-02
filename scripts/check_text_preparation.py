#!/usr/bin/env python3
"""Verify host-free text preparation against saved official scoring fixtures."""
import hashlib,json,pathlib
from prepare_text import TextPreparer
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=TextPreparer();cases=json.loads((ROOT/'benchmarks/cases.json').read_text());refs=json.loads((ROOT/'artifacts/reference-orders.json').read_text())['records'];rows=[]
for r in refs:
 c=next(c for c in cases if c['id']==r['id']);a=p.prepare({**c,'options':r['options']})
 assert a['token_ids']==r['token_ids'] and a['input_sha256']==r['input_sha256'],r['id']
 rows.append(dict(id=r['id'],offset=r['offset'],tokens=len(a['token_ids']),input_sha256=a['input_sha256'],exact_match=True))
report=dict(model_lock_sha256=hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest(),records=rows)
(ROOT/'docs/text-preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{len(rows)} official text fixtures match exactly')
