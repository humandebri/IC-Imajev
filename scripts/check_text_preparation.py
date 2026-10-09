#!/usr/bin/env python3
"""Verify the shorter default against saved prompts and preserve evidence text."""
import hashlib,json,pathlib
from prepare_text import IMAGE_EVIDENCE_INSTRUCTION, UNKNOWN_INSTRUCTION
from prepare_text_legacy import TextPreparer, PROMPT_LAYOUT
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=TextPreparer();cases=json.loads((ROOT/'benchmarks/cases.json').read_text());refs=json.loads((ROOT/'artifacts/reference-orders.json').read_text())['records'];rows=[]
for r in refs:
 c=next(c for c in cases if c['id']==r['id']);a=p.prepare({**c,'options':r['options']})
 original=p.render(r['prompt'])
 assert p.tokenizer.encode(original,add_special_tokens=False)==r['token_ids'],r['id']
 old_header,sep,body=r['prompt'].partition('\nState: ')
 expected_header=old_header.replace(IMAGE_EVIDENCE_INSTRUCTION+' ','').replace(UNKNOWN_INSTRUCTION+' ','')
 old_unknown='\n'+chr(65+len(r['options']))+': unknown — cannot be determined from the available evidence, the premise is false, or no listed option is correct'
 assert body.endswith(old_unknown),r['id']
 expected=p.render(expected_header+sep+body[:-len(old_unknown)]+'\n'+chr(65+len(r['options']))+': unknown')
 assert a['token_ids']==p.tokenizer.encode(expected,add_special_tokens=False),r['id']
 assert a['input_sha256']==hashlib.sha256(json.dumps(a['token_ids']).encode()).hexdigest(),r['id']
 assert p.tokenizer.decode(a['token_ids'][:a['prefix_tokens']],skip_special_tokens=False).endswith('State: "' if isinstance(c['state'],str) else 'State: '),r['id']
 rows.append(dict(id=r['id'],offset=r['offset'],tokens=len(a['token_ids']),original_tokens=len(r['token_ids']),prefix_tokens=a['prefix_tokens'],input_sha256=a['input_sha256'],matches_edited_prompt=True))
# Repeated instruction/unknown wording in evidence and user options must survive.
sentinel=IMAGE_EVIDENCE_INSTRUCTION+' '+UNKNOWN_INSTRUCTION+' unknown — cannot be determined from the available evidence, the premise is false, or no listed option is correct'
for state in [sentinel,{'text':sentinel},'',{}]:
 a=p.prepare(dict(id='preserve-evidence',state=state,question=sentinel,options=[dict(value='sentinel',description=sentinel),'other']))
 assert a['prompt'].count(sentinel)==(3 if state else 2)
 assert a['prompt'].endswith('\nC: unknown')
 prefix=a['token_ids'][:a['prefix_tokens']]
 assert prefix and p.tokenizer.encode(p.render(a['prompt']),add_special_tokens=False)[:len(prefix)]==prefix
report=dict(model_lock_sha256=hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest(),prompt_layout=PROMPT_LAYOUT,scope='Input preparation only; no inference accuracy or output parity claim',records=rows)
(ROOT/'docs/text-preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(f'{len(rows)} text-only inputs match the edited saved prompts')
