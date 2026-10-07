#!/usr/bin/env python3
"""Exact-input prefix partitioning and measured query fusion for proposal tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_prompt_accuracy import base_flags, MODULE
D=ROOT/'artifacts/proposal-query-optimization-v1'
SOURCE=ROOT/'artifacts/proposal-assessment-500-660-20261006/evaluation'
CANDIDATE='6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def prepare():
 D.mkdir(exist_ok=False)
 fixture=json.loads((SOURCE/'inputs.json').read_text());records=fixture['records']
 shutil_source=(ROOT/'scripts/run_merged_query32.py').read_text()
 shutil_source=shutil_source.replace('ROOT=pathlib.Path(__file__).resolve().parents[1]',f'ROOT=pathlib.Path({str(ROOT)!r})')
 shutil_source=shutil_source.replace('allowed_prefixes=(27,38)','allowed_prefixes=tuple(range(1,39))')
 shutil_source=shutil_source.replace('import query32_templates','import proposal_query32_graph as query32_templates')
 shutil_source=shutil_source.replace('from query32_templates import Query32PrefixGraph','from proposal_query32_graph import Query32PrefixGraph')
 (D/'runner.py').write_text(shutil_source)
 graph=(ROOT/'client/query32_templates.py').read_text().replace('p not in (27,38)','not 1<=p<=38')
 graph=graph.replace("MODULE='2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05'",f"MODULE='{CANDIDATE}'")
 (D/'proposal_query32_graph.py').write_text(graph)
 # Fixed before executing: exact prefixes, no altered tokens/prompts/weights.
 banks=[];plans=[]
 for i in sorted(range(len(records)),key=lambda i:-len(records[i]['token_ids'])):
  ids=records[i]['token_ids'];route='query32' if len(ids)<=97 else 'tail';cap=59 if route=='query32' else 80
  b=next((j for j,x in enumerate(banks) if x['route']==route and ids[:len(x['ids'])]==x['ids'] and 1<=len(ids)-len(x['ids'])<=cap),None)
  if b is None:b=len(banks);banks.append(dict(route=route,ids=ids[:len(ids)-cap],record=i))
  plans.append(dict(record=i,bank=b,prefix=len(banks[b]['ids']),suffix=len(ids)-len(banks[b]['ids']),route=route))
 save(D/'inputs.json',fixture);save(D/'plan.json',dict(banks=banks,records=sorted(plans,key=lambda r:r['record']),source_inputs_sha256=sha(SOURCE/'inputs.json'),
      scope='Exact task token sequences retained. Prefix computation is input-dependent and separately charged; it is not a universal free fixed prompt cache.'))
 paths=[ROOT/'scripts/run_prefix_canister.py',ROOT/'scripts/prepare_prefix_reuse.py',ROOT/'scripts/evaluate_prompt_accuracy.py',Path(__file__),D/'runner.py',D/'proposal_query32_graph.py']+list((ROOT/'client').glob('*.py'))
 save(D/'source-hashes.json',{str(p):sha(p) for p in paths})
 print(json.dumps(dict(banks=len(banks),records=len(records))),flush=True)

def command(candidate=False):
 cmd=base_flags();cmd[1]=str(D/'runner.py' if candidate else ROOT/'scripts/run_prefix_canister.py')
 cmd[cmd.index('--reference')+1]=str(D/'inputs.json')
 if candidate:
  for flag,value in [('--canister','2hlkr-gl777-77775-aaaxa-cai'),('--wasm','artifacts/merged-query32-v1/build/full.wasm'),('--bridge-binary','artifacts/query32-v1/client-build/imajev-client')]:cmd[cmd.index(flag)+1]=value
 return cmd

def call(cmd,dest):
 dest.mkdir(parents=True,exist_ok=True);save(dest/'command.json',cmd)
 with (dest/'run.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log,check=True)
 return json.loads((dest/'report.json').read_text())

def bank(number):
 p=json.loads((D/'plan.json').read_text())['banks'][number];dest=D/'prefixes'/f'{number:02d}';cache=dest/'queries';packets=D/'packets'/f'{number:02d}'
 if not (dest/'report.json').exists():
  cmd=command();cmd[cmd.index('--cache')+1]=str(cache)
  print(json.dumps(dict(stage='prefix-start',bank=number,tokens=len(p['ids']))),flush=True)
  call(cmd+['--record',str(p['record']),'--prefix-tokens',str(len(p['ids'])),'--prepare-prefix','--directory',str(dest)],dest)
 r=json.loads((dest/'report.json').read_text());assert r['tokens']==len(p['ids']) and not r['replayed_queries']
 if not (packets/'cache.json').exists():
  packets.parent.mkdir(exist_ok=True)
  cmd=[sys.executable,'-B',str(ROOT/'scripts/prepare_prefix_reuse.py'),'--prefix-directory',str(dest),'--directory',str(packets),'--canister','2vn5i-k3777-77775-aaaua-cai','--codec-module','e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e','--run-report',str(dest/'codec-report.json')]
  with (dest/'codec.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log,check=True)
 print(json.dumps(dict(stage='prefix-ready',bank=number,queries=r['query_count'])),flush=True)
 return cache,packets

def validate(i,dest):
 import struct
 old=json.loads((SOURCE/'runs'/f'{i:03d}'/'report.json').read_text());new=json.loads((dest/'report.json').read_text())
 assert new['input_hash']==old['input_hash'] and new['model']==old['model'] and new['pack_hash']==old['pack_hash']
 assert not new['replayed_queries'] and new['executed_query_count']==new['query_count']
 assert new['wasm_sha256']==new['deployed_wasm_sha256'] in (MODULE,CANDIDATE)
 assert new['max_query_instructions']<5_000_000_000 and new['max_observed_heap_bytes']<2**32
 a=old['decision_query']['ok']['decision'];b=new['decision_query']['ok']['decision']
 assert all(a[k]==b[k] for k in ('value','abstained','raw_logits','probabilities','unknown_probability'))
 hashes=json.loads((SOURCE/'runs'/f'{i:03d}'/'staging-intermediate-artifact-hashes.json').read_text())
 assert sha(dest/'final-hidden.npy')==hashes['final-hidden.npy']['sha256'],'final normalized hidden differs'
 row=dict(record=i,baseline_queries=old['query_count'],queries=new['query_count'],baseline_instructions=old['total_instructions'],instructions=new['total_instructions'],baseline_candid_bytes=old['total_candid_bytes'],candid_bytes=new['total_candid_bytes'],
  input_sha256=new['input_hash'],report_sha256=sha(dest/'report.json'),raw_logits_bit_equal=True,decision_equal=True,final_hidden_bit_equal=True,route=new.get('query32_effective',False) and 'query32' or 'fused-prefix',
  fallback=bool(new.get('fallback')),seconds=new['wall_seconds_this_run'])
 save(dest/'verified.json',row)
 return row

def execute(only=None):
 for path,h in json.loads((D/'source-hashes.json').read_text()).items():assert sha(Path(path))==h,f'changed source {path}'
 plan=json.loads((D/'plan.json').read_text())
 for p in plan['records']:
  i=p['record']
  if only is not None and i!=only:continue
  dest=D/'runs'/f'{i:03d}'
  if (dest/'verified.json').exists():continue
  cache,packets=bank(p['bank']);candidate=p['route']=='query32'
  cmd=command(candidate);cmd[cmd.index('--cache')+1]=str(cache)
  flags=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start']
  if candidate:flags+=['--query32-start','--step-offset',str(1000000+i*10000)]
  else:
   # Use packed schedule only inside its existing checked module/input bounds.
   if p['suffix']<=69 and p['prefix']<=27:flags+=['--packed-start']
  print(json.dumps(dict(stage='inference-start',**p)),flush=True)
  call(cmd+flags+['--record',str(i),'--directory',str(dest)],dest)
  row=validate(i,dest);print(json.dumps(dict(stage='verified',**row)),flush=True)
  summarize()
 summarize()

def summarize():
 plan=json.loads((D/'plan.json').read_text());rows=[];preparations=[]
 for p in plan['records']:
  path=D/'runs'/f"{p['record']:03d}"/'verified.json'
  if path.exists():rows.append(json.loads(path.read_text()))
 for b in range(len(plan['banks'])):
  d=D/'prefixes'/f'{b:02d}'
  if (d/'codec-report.json').exists():
   r=json.loads((d/'report.json').read_text());c=json.loads((d/'codec-report.json').read_text())
   preparations.append(dict(bank=b,queries=r['query_count'],instructions=r['total_instructions'],candid_bytes=r['total_candid_bytes'],codec=c))
 save(D/'summary.json',dict(complete=len(rows)==len(plan['records']),rows=rows,preparations=preparations))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--record',type=int);a=p.parse_args()
 if a.mode=='prepare':prepare()
 else:execute(a.record)
