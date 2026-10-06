#!/usr/bin/env python3
"""Offline routing check: new values/questions are outside the prefix cache."""
import hashlib,json,sys
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from template_prefix import select,MODULE,VOTING38

def main():
 d=ROOT/'artifacts/voting-template-prefix-v1/selection-check';d.mkdir(exist_ok=False)
 fixture=json.loads((ROOT/'artifacts/text-short-v2/inputs.json').read_text());t=Tokenizer.from_file(str(ROOT/'checkpoints/base/tokenizer.json'))
 text=t.decode(fixture['records'][0]['token_ids'],skip_special_tokens=False)
 assert t.encode(text,add_special_tokens=False).ids==fixture['records'][0]['token_ids']
 rows=[]
 for value in ['0','1','2','7','42','365','20000','999999','1000000']:
  for question in ['Could this change reduce voter participation or concentrate voting power?','Does this change increase the required voting lock duration?']:
   prompt=text.replace('20000',value).replace('Could this change reduce voter participation or concentrate voting power?',question)
   ids=t.encode(prompt,add_special_tokens=False).ids;assert tuple(ids[:38])==VOTING38
   a,b,name=select(ROOT,ids,MODULE,'common','packets');assert name=='voting-fixed-stem38'
   rows.append(dict(value=value,question=question,tokens=len(ids),prefix=38,suffix=len(ids)-38))
 for index,name in [(0,'voting-fixed-stem38'),(1,'voting-fixed-stem38'),(2,'common')]:assert select(ROOT,fixture['records'][index]['token_ids'],MODULE,'common','packets')[2]==name
 assert select(ROOT,fixture['records'][0]['token_ids'],'6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931','common','packets')[2]=='common'
 p=text.replace('from 1 day','from 3 days');assert select(ROOT,t.encode(p,add_special_tokens=False).ids,MODULE,'common','packets')[2]=='common'
 report=dict(scope=__doc__,cases=rows,network_queries=0,selection_source_sha256=hashlib.sha256((ROOT/'client/template_prefix.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),passed=True)))
if __name__=='__main__':main()
