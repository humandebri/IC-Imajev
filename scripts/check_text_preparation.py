#!/usr/bin/env python3
"""Verify text-only preparation against the saved prompts with one instruction edit."""
import hashlib,json,pathlib
from prepare_text import TextPreparer, PROMPT_LAYOUT, IMAGE_EVIDENCE_INSTRUCTION, TEXT_EVIDENCE_INSTRUCTION
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=TextPreparer();cases=json.loads((ROOT/'benchmarks/cases.json').read_text());refs=json.loads((ROOT/'artifacts/reference-orders.json').read_text())['records'];rows=[]
for r in refs:
 c=next(c for c in cases if c['id']==r['id']);a=p.prepare({**c,'options':r['options']})
 original=p.render(r['prompt'])
 assert p.tokenizer.encode(original,add_special_tokens=False)==r['token_ids'],r['id']
 expected=original.replace(IMAGE_EVIDENCE_INSTRUCTION,TEXT_EVIDENCE_INSTRUCTION,1)
 assert a['token_ids']==p.tokenizer.encode(expected,add_special_tokens=False),r['id']
 assert a['input_sha256']==hashlib.sha256(json.dumps(a['token_ids']).encode()).hexdigest(),r['id']
 assert p.tokenizer.decode(a['token_ids'][:a['prefix_tokens']],skip_special_tokens=False).endswith('State: "' if isinstance(c['state'],str) else 'State: '),r['id']
 rows.append(dict(id=r['id'],offset=r['offset'],tokens=len(a['token_ids']),original_tokens=len(r['token_ids']),prefix_tokens=a['prefix_tokens'],input_sha256=a['input_sha256'],matches_edited_prompt=True))
report=dict(model_lock_sha256=hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest(),prompt_layout=PROMPT_LAYOUT,scope='Input preparation only; no inference accuracy or output parity claim',records=rows)
(ROOT/'docs/text-preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{len(rows)} text-only inputs match the edited saved prompts')
