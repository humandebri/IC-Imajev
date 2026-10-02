#!/usr/bin/env python3
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
src=pathlib.Path('/Volumes/KINGSTON/ICP/IC-Laya-Standalone/artifacts/boomdao_query_benchmark/summary.json')
baseline=json.loads(src.read_text())
(root/'benchmarks/laya_baseline.json').write_text(json.dumps({'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'baseline':baseline},indent=2)+'\n')
cases=[]
for c in baseline['cases']:
    cases.append({'id':f"historical-{c['proposal_id']}",'split':'diagnostic','kind':'historical','proposal_id':c['proposal_id'],'question':c['question'],'options':c['options'],'state':c['state'],'gold':None,'reason':'Historical risk question has no uniquely adjudicated ground truth; compare labels only.'})
rows=[
('minimum','The minimum dissolve delay needed to vote changes from 1 day to 20000 days. Neurons below the new minimum cannot vote.','Does this change raise the minimum lock duration needed for voting?','yes','Explicit voting eligibility rule and a 20000-fold increase.'),
('maximum','Maximum permitted dissolve delay increases from 2629800 seconds to 2629800000 seconds. Minimum voting delay is unchanged.','Does this proposal increase the maximum permitted lock duration?','yes','Explicit maximum increases 1000-fold.'),
('stake','Minimum stake needed to create a neuron increases from 10 to 10000 tokens.','Does this proposal raise the minimum stake required to create a neuron?','yes','Explicit minimum stake increases 1000-fold.'),
('mint','The proposal mints 250000000 new tokens to one account. Existing total supply is not provided.','Does this proposal create new tokens?','yes','Explicit mint; concentration relative to existing supply is a separate unknown.'),
('unchanged','Minimum voting delay was 86400 seconds and remains 86400 seconds. No other parameter changes.','Does this proposal increase the minimum lock duration needed for voting?','no','Identical old/new values; tests false alerts.'),
('insufficient','The proposal specifies a new minimum voting delay of 86400 seconds. The old minimum is not provided.','Does this proposal increase the minimum lock duration needed for voting?','__unknown__','Cannot establish direction without old value.'),
('maximum-not-minimum','Maximum permitted lock duration increases 1000-fold. Minimum voting delay stays 1 day.','Does this proposal increase the minimum lock duration needed for voting?','no','Maximum and minimum are distinct; tests false alerts.'),
]
for ident,state,q,gold,reason in rows:
    cases.append({'id':ident,'split':'evaluation','kind':'controlled','question':q,'options':['yes','no'],'state':state,'gold':gold,'reason':reason})
(root/'benchmarks/cases.json').write_text(json.dumps(cases,indent=2)+'\n')
