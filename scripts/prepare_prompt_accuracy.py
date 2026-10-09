#!/usr/bin/env python3
"""Freeze paired original/current prompts and explicit golds before inference."""
import hashlib,json,pathlib
from prepare_text_legacy import TextPreparer
from vision_decision.contracts import ChoiceField,Option
from vision_decision.scoring import compile_question
ROOT=pathlib.Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/prompt-accuracy-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=TextPreparer();base=[c for c in json.loads((ROOT/'benchmarks/cases.json').read_text()) if c['gold']is not None]
 for c in base:c['category']='existing-control'
 cases=base+json.loads((ROOT/'benchmarks/prompt_accuracy_extra.json').read_text())
 assert len(cases)==24 and len({c['id'] for c in cases})==24
 swaps={'minimum','maximum','insufficient','delay-equal-units','false-premise-mint','no-correct-delay-option','embedded-yes-override','direction-three-way'}
 records=[];pairs=[]
 original_prefix=json.loads((ROOT/'artifacts/single-quad/full-proof-v2/prefix/queries/cache.json').read_text())['token_ids']
 short_prefix=json.loads((ROOT/'artifacts/query-packing-v3/prefix-v2/queries/cache.json').read_text())['token_ids']
 for case in cases:
  for offset in range(2 if case['id'] in swaps else 1):
   c=dict(case,options=case['options'][offset:]+case['options'][:offset]);current=p.prepare(c)
   field=ChoiceField(id=c['id'].replace('-','_'),type='choice',question=c['question'],options=[Option(value=x) for x in c['options']])
   header,choices,texts=compile_question(field,c['state'],'standard');labels=[x['code'] for x in p.binding['codes'][:len(choices)]]
   prompt=header+'\n'.join(f'{label}: {text}' for label,text in zip(labels,texts));ids=p.tokenizer.encode(p.render(prompt),add_special_tokens=False)
   original=dict(current,prompt=prompt,token_ids=ids,input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),prompt_layout='official-standard',prefix_tokens=len(original_prefix))
   assert ids[:len(original_prefix)]==original_prefix,c['id']
   assert current['token_ids'][:len(short_prefix)]==short_prefix and current['prefix_tokens']==len(short_prefix),c['id']
   assert len(ids)-len(current['token_ids'])==38,c['id']
   pair=dict(id=c['id'],offset=offset,category=c['category'],gold=c['gold'],reason=c['reason'],options=c['options'])
   for variant,record in [('original',original),('short',current)]:
    pair[variant+'_record']=len(records);records.append(dict(record,variant=variant,offset=offset,category=c['category']))
   pairs.append(pair)
 assert len(pairs)==32
 out=dict(model_lock_sha256=sha(ROOT/'MODEL_LOCK.json'),records=records,pairs=pairs,distinct_cases=24,source_hashes={str(q.relative_to(ROOT)):sha(q) for q in [ROOT/'benchmarks/cases.json',ROOT/'benchmarks/prompt_accuracy_extra.json',ROOT/'scripts/prepare_text.py',ROOT/'scripts/prepare_text_legacy.py',pathlib.Path(__file__)]})
 path=D/'inputs.json'
 if path.exists():raise RuntimeError('refusing to change frozen evaluation inputs')
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(distinct_cases=24,paired_orders=32,inference_runs=len(records),min_tokens=min(len(r['token_ids']) for r in records),max_tokens=max(len(r['token_ids']) for r in records),max_short_suffix=max(len(r['token_ids'])-r['prefix_tokens'] for r in records if r['variant']=='short'))))
if __name__=='__main__':main()
