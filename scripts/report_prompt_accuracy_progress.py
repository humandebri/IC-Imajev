#!/usr/bin/env python3
"""Aggregate completed independent journals without counting an unfinished pair."""
import json
import evaluate_prompt_accuracy as e
D=e.ROOT/'artifacts/prompt-accuracy-v2';f=json.loads((D/'inputs.json').read_text());rows=[]
for p in f['pairs']:
 for v in ['original','short']:
  d=D/'runs'/f'{p["id"]}-o{p["offset"]}-{v}'
  if (d/'report.json').exists():rows.append(e.extract(json.loads((d/'report.json').read_text()),f['records'][p[v+'_record']],p,v,d))
s=e.summarize(f['pairs'],rows);e.atomic_json(D/'progress.json',dict(completed_inferences=len(rows),**s))
print(json.dumps(dict(completed_inferences=len(rows),completed_pairs=len(s['paired_comparisons']),regressions=s['regressions'],improvements=s['improvements'],answer_changes=s['answer_changes']),ensure_ascii=False))
