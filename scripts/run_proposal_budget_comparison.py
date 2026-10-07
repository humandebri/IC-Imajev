#!/usr/bin/env python3
"""Fresh same-model paired state-compression regression measurement."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/proposal-full-161-20261007/budget-comparison'
sys.path.insert(0,str(ROOT/'scripts'))
from evaluation_run_lock import run_lock
MODULE='6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'
CODEC='e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);pending=p.with_suffix(p.suffix+'.pending');pending.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');pending.replace(p)
def frozen():
 for name in ('identities.json','execution-identities.json'):
  for path,h in json.loads((D/name).read_text()).items():assert sha(Path(path))==h,f'source changed: {path}'
def call(cmd,dest):
 dest.mkdir(parents=True,exist_ok=True);save(dest/'command.json',cmd)
 with (dest/'run.log').open('a') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=f,check=True)
 frozen()
def setup():
 paths=[Path(__file__),ROOT/'artifacts/proposal-query-optimization-v1/same-module-runner.py',ROOT/'artifacts/proposal-query-optimization-v1/proposal_query32_graph.py',ROOT/'scripts/run_merged_query32.py',ROOT/'scripts/prepare_prefix_reuse.py',ROOT/'checkpoints/full-int8.manifest.json',ROOT/'artifacts/merged-query32-v1/build/full.wasm',ROOT/'artifacts/query32-v1/client-build/imajev-client']+list((ROOT/'client').glob('*.py'))
 value={str(p):sha(p) for p in paths};identity=D/'execution-identities.json'
 if identity.exists():assert json.loads(identity.read_text())==value
 else:save(identity,value)
 frozen()
 cmd=json.loads((ROOT/'artifacts/proposal-full-161-20261007/base-command.json').read_text())
 cmd[1]=str(ROOT/'artifacts/proposal-query-optimization-v1/same-module-runner.py')
 for flag,value in [('--reference',str(D/'inputs.json')),('--cache',str(D/'prefix/queries'))]:cmd[cmd.index(flag)+1]=value
 return cmd
def validate(i):
 f=json.loads((D/'inputs.json').read_text());x=f['records'][i];dest=D/'runs'/f'{i:03d}';r=json.loads((dest/'report.json').read_text())
 assert r['input_hash']==x['input_sha256'] and r['model']==f['model_lock_sha256']
 assert r['pack_hash']==json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())['pack_hash']
 assert r['wasm_sha256']==r['deployed_wasm_sha256']==MODULE
 assert r['tokens']==len(x['token_ids']) and r['query_count']==r['executed_query_count']
 assert not r['replayed_queries'] and not r.get('fallback')
 assert r['comparison']['typed_output_valid'] and [z['layer'] for z in r['layers']]==list(range(32))
 assert r['max_query_instructions']<5_000_000_000 and r['max_observed_heap_bytes']<2**32
 assert all(q['ok']['request_bytes']<2_000_000 and q['ok']['reply_bytes']<2_000_000 for q in r['queries'])
 if x['variant']=='compact':assert r['query32_effective'] and r['query_count']==32 and r['tokens']<=86
 else:assert r['tokens']<=128
 d=r['decision_query']['ok']['decision'];prediction='__unknown__' if d['abstained'] else d['value']
 assert len(d['raw_logits'])==len(x['options'])+1
 row={'record':i,'id':x['id'],'variant':x['variant'],'offset':x['offset'],'tokens':r['tokens'],'gold':x['gold'],'prediction':prediction,'correct':prediction==x['gold'],
      'raw_logits':d['raw_logits'],'probabilities':d['probabilities'],'unknown_probability':d['unknown_probability'],
      'queries':r['query_count'],'instructions':r['total_instructions'],'candid_bytes':r['total_candid_bytes'],'seconds':r['wall_seconds_this_run'],'report_sha256':sha(dest/'report.json')}
 save(dest/'verified.json',row)
 hashes={str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (dest/'queries').rglob('*') if p.is_file() and p.suffix in ('.bin','.npy','.npz','.bf16')}
 save(dest/'intermediate-artifact-hashes.json',hashes)
 for path in hashes:(dest/path).unlink()
 return row
def metrics(rows):
 return {'total':len(rows),'correct':sum(x['correct'] for x in rows),'accuracy':sum(x['correct'] for x in rows)/len(rows) if rows else None,
         'unsupported_answers':sum(x['gold']=='__unknown__' and x['prediction']!='__unknown__' for x in rows),
         'false_abstentions':sum(x['gold']!='__unknown__' and x['prediction']=='__unknown__' for x in rows)}
def summarize():
 f=json.loads((D/'inputs.json').read_text());p=json.loads((D/'prepared.json').read_text());rows=[]
 for i in range(len(f['records'])):
  path=D/'runs'/f'{i:03d}'/'verified.json'
  if path.exists():
   row=json.loads(path.read_text());assert row['report_sha256']==sha(path.parent/'report.json');rows.append(row)
 lookup={x['record']:x for x in rows};paired=[]
 for c in p['pairs']:
  a=lookup.get(c['full_record']);b=lookup.get(c['compact_record'])
  if a is None or b is None:continue
  paired.append({'id':c['id'],'offset':c['offset'],'category':c['category'],'gold':c['gold'],'full_prediction':a['prediction'],'compact_prediction':b['prediction'],
                 'full_correct':a['correct'],'compact_correct':b['correct'],'regression':a['correct'] and not b['correct'],'improvement':not a['correct'] and b['correct'],
                 'answer_changed':a['prediction']!=b['prediction'],'full_tokens':a['tokens'],'compact_tokens':b['tokens'],'full_queries':a['queries'],'compact_queries':b['queries']})
 variants={v:metrics([x for x in rows if x['variant']==v and x['offset']==0]) for v in ('full','compact')}
 order=[]
 for c in p['pairs']:
  if c['offset']!=1:continue
  for v in ('full','compact'):
   match={x['offset']:x for x in rows if x['variant']==v and x['id']==c['id']}
   if set(match)!={0,1}:continue
   order.append({'id':c['id'],'variant':v,'changed':match[0]['prediction']!=match[1]['prediction'],'base':match[0]['prediction'],'reordered':match[1]['prediction']})
 r={'complete':len(rows)==64,'completed':len(rows),'planned':64,'distinct_cases':24,'metrics':variants,
    'all_ordered_metrics':{v:metrics([x for x in rows if x['variant']==v]) for v in ('full','compact')},
    'regressions':[x for x in paired if x['regression']],'improvements':[x for x in paired if x['improvement']],
    'order_changes':[x for x in order if x['changed']],'paired':paired,'runs':rows,
    'development_set':True,'proposal_161_accuracy_measured':False,
    'inference_queries':sum(x['queries'] for x in rows),'inference_instructions':sum(x['instructions'] for x in rows)}
 save(D/'report.json',r)
 lines=['# 128-token上限と86-token上限：証拠保持の比較','',
        '2026-10-07。既存の明示的証拠による24ケース、選択肢順序違いを含む32組・64推論。質問・選択肢・モデルを固定し、stateの表現だけ短縮。数値と単位は正確に保持（最大lockのみseconds→minutesの厳密換算）。原文の否定・未知値・対立する証拠・引用された攻撃指示を保持。', '',
        f"完了 {len(rows)}/64。主要24ケース: {variants}。",'',
        '開発用対照ケースの正答率であり、161件のproposal賛否の正答率ではない。上限128側の実入力は71〜96 tokens、86側は67〜85 tokens。128 tokensを満杯にした比較でも、未知proposalの一般化性能の証明でもない。', '',
        f"回帰{len(r['regressions'])}、改善{len(r['improvements'])}、選択肢順序で出力が変わった例{len(r['order_changes'])}（すべての順序組を別計上）。", '',
        '|case|順序|正解|元の出力|短縮後|元tokens|短縮tokens|元query|短縮query|', '|---|---:|---|---|---|---:|---:|---:|---:|']
 for x in paired:lines.append(f"|{x['id']}|{x['offset']}|{x['gold']}|{x['full_prediction']}|{x['compact_prediction']}|{x['full_tokens']}|{x['compact_tokens']}|{x['full_queries']}|{x['compact_queries']}|")
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
 return r
def execute(limit=None):
 base=setup();f=json.loads((D/'inputs.json').read_text());prefix=D/'prefix';packets=D/'packets'
 if not (prefix/'report.json').exists():call(base+['--prepare-prefix','--prefix-tokens','27','--record','0','--directory',str(prefix)],prefix)
 if not (packets/'cache.json').exists():call([sys.executable,'-B',str(ROOT/'scripts/prepare_prefix_reuse.py'),'--prefix-directory',str(prefix),'--directory',str(packets),'--canister','7vs54-wt777-77775-aaajq-cai','--codec-module',CODEC,'--run-report',str(packets/'report.json')],packets)
 flags=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start']
 n=0
 for i,x in enumerate(f['records']):
  dest=D/'runs'/f'{i:03d}'
  if (dest/'verified.json').exists():continue
  if limit is not None and n>=limit:break
  route=['--query32-start'] if len(x['token_ids'])-27<=59 else []
  print(json.dumps({'stage':'start','record':i,'case':x['id'],'variant':x['variant'],'tokens':len(x['token_ids'])}),flush=True)
  call(base+flags+route+['--record',str(i),'--step-offset',str(14000000+i*10000),'--directory',str(dest)],dest)
  row=validate(i);n+=1;summarize();print(json.dumps({'stage':'verified',**row}),flush=True)
 print(json.dumps({k:v for k,v in summarize().items() if k not in ('runs','paired')}),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['run','report']);ap.add_argument('--limit',type=int);args=ap.parse_args()
 with run_lock(D):
  if args.mode=='run':execute(args.limit)
  else:print(json.dumps({k:v for k,v in summarize().items() if k not in ('runs','paired')}),flush=True)
