#!/usr/bin/env python3
"""Actual local-canister paired evaluation of the original and short prompt."""
import argparse,ast,hashlib,json,pathlib,subprocess,sys,time
from evaluation_run_lock import run_lock
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
CODEC='e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def base_flags():
 tree=ast.parse((ROOT/'artifacts/text-short-v2/run_queries.py').read_text());node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='base' for t in n.targets))
 cmd=eval(compile(ast.Expression(node),'flags','eval'),{'R':ROOT,'sys':sys})
 cmd[cmd.index('--wasm')+1]='artifacts/query-packing-v3/build/full.wasm';cmd[cmd.index('--reference')+1]='artifacts/prompt-accuracy-v2/inputs.json'
 cmd+=['--bridge-binary','artifacts/query-packing-v3/build/imajev-client']
 return cmd
def extract(report,record,pair,variant,directory):
 d=report['decision_query']['ok']['decision'];pred='__unknown__' if d['abstained'] else d['value']
 assert report['wasm_sha256']==MODULE and report['input_hash']==record['input_sha256']
 assert report['comparison']['typed_output_valid'] and len(d['probabilities'])==len(record['options'])
 assert report['comparison']['gold']==pair['gold']
 gold_probability=d['unknown_probability'] if pair['gold']=='__unknown__' else d['probabilities'][record['options'].index(pair['gold'])]
 return dict(id=pair['id'],offset=pair['offset'],category=pair['category'],variant=variant,gold=pair['gold'],prediction=pred,correct=pred==pair['gold'],gold_probability=gold_probability,unknown_probability=d['unknown_probability'],probabilities=d['probabilities'],options=record['options'],tokens=report['tokens'],suffix_tokens=report['processed_tokens'],query_count=report['query_count'],fallback=bool(report.get('fallback')),replayed_queries=report['replayed_queries'],report=str((directory/'report.json').relative_to(ROOT)),report_sha256=sha(directory/'report.json'))
def metrics(rows):
 known=[r for r in rows if r['gold']!='__unknown__'];unknown=[r for r in rows if r['gold']=='__unknown__'];answered=[r for r in rows if r['prediction']!='__unknown__']
 return dict(total=len(rows),correct=sum(r['correct'] for r in rows),accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,known_total=len(known),known_correct=sum(r['correct'] for r in known),unknown_total=len(unknown),unknown_correct=sum(r['correct'] for r in unknown),false_abstentions=sum(r['prediction']=='__unknown__' for r in known),unsupported_answers=sum(r['prediction']!='__unknown__' for r in unknown),answered_total=len(answered),answered_correct=sum(r['correct'] for r in answered),mean_gold_probability=sum(r['gold_probability'] for r in rows)/len(rows) if rows else None)
def summarize(pairs,rows):
 paired=[]
 for p in pairs:
  found={r['variant']:r for r in rows if r['id']==p['id'] and r['offset']==p['offset']}
  if len(found)!=2:continue
  a,b=found['original'],found['short'];paired.append(dict(id=p['id'],offset=p['offset'],category=p['category'],gold=p['gold'],original_prediction=a['prediction'],short_prediction=b['prediction'],original_correct=a['correct'],short_correct=b['correct'],changed=a['prediction']!=b['prediction'],regression=a['correct'] and not b['correct'],improvement=not a['correct'] and b['correct'],gold_probability_change=b['gold_probability']-a['gold_probability']))
 by_variant={v:metrics([r for r in rows if r['variant']==v and r['offset']==0]) for v in ['original','short']}
 categories={c:{v:metrics([r for r in rows if r['category']==c and r['variant']==v and r['offset']==0]) for v in ['original','short']} for c in sorted({r['category'] for r in rows})}
 order_checks=[]
 for p in pairs:
  if p['offset']!=1:continue
  for v in ['original','short']:
   found={r['offset']:r for r in rows if r['id']==p['id'] and r['variant']==v}
   if len(found)!=2:continue
   a,b=found[0],found[1];order_checks.append(dict(id=p['id'],variant=v,base_prediction=a['prediction'],reordered_prediction=b['prediction'],changed=a['prediction']!=b['prediction'],base_correct=a['correct'],reordered_correct=b['correct']))
 return dict(primary_distinct_cases=by_variant,all_ordered_cases={v:metrics([r for r in rows if r['variant']==v]) for v in ['original','short']},by_category=categories,paired_comparisons=paired,regressions=[r for r in paired if r['regression']],improvements=[r for r in paired if r['improvement']],answer_changes=[r for r in paired if r['changed']],order_checks=order_checks,order_changes=[r for r in order_checks if r['changed']])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/prompt-accuracy-v2');args=ap.parse_args();d=ROOT/args.directory;d.mkdir(parents=True,exist_ok=True)
 fixture_path=ROOT/'artifacts/prompt-accuracy-v2/inputs.json';fixture=json.loads(fixture_path.read_text());base=base_flags()
 paths=[fixture_path,ROOT/'MODEL_LOCK.json',ROOT/'artifacts/query-packing-v3/build/full.wasm',ROOT/'artifacts/query-packing-v3/build/imajev-client',ROOT/'client/packed_inference.py',ROOT/'scripts/run_prefix_canister.py',pathlib.Path(__file__),ROOT/'scripts/evaluation_run_lock.py',ROOT/'artifacts/query-packing-v3/packets-v2/cache.json']
 identities={str(p.relative_to(ROOT)):sha(p) for p in paths}
 assert sha(ROOT/'artifacts/query-packing-v3/build/full.wasm')==MODULE
 session=dict(scope='Actual local canister INT8 inference; paired prompts, unchanged weights/readout/calibration; curated explicit golds',module=MODULE,distinct_gold_cases=24,paired_ordered_cases=32,planned_inferences=64,rotations_per_inference=1,source_hashes=identities)
 sp=d/'evaluation-session.json'
 if sp.exists():assert json.loads(sp.read_text())==session,'evaluation session changed'
 else:atomic_json(sp,session)
 def run(cmd,log):
  with log.open('a') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 prefix=d/'original-prefix';cache=prefix/'queries';packets=d/'original-packets'
 prep=list(base);prep[prep.index('--cache')+1]=str(cache)
 if not (prefix/'report.json').exists():run(prep+['--record','0','--prefix-tokens','45','--prepare-prefix','--directory',str(prefix)],d/'prefix-preparation.log')
 if not (packets/'cache.json').exists():run([sys.executable,str(ROOT/'scripts/prepare_prefix_reuse.py'),'--prefix-directory',str(prefix),'--directory',str(packets),'--canister','2vn5i-k3777-77775-aaaua-cai','--codec-module',CODEC,'--run-report',str(d/'original-packet-preparation.json')],d/'packet-preparation.log')
 rows=[];started=time.monotonic()
 for pair in fixture['pairs']:
  for variant in ['original','short']:
   index=pair[variant+'_record'];record=fixture['records'][index];dest=d/'runs'/f'{pair["id"]}-o{pair["offset"]}-{variant}'
   path=dest/'report.json';cmd=list(base);cmd[cmd.index('--cache')+1]=str(cache if variant=='original' else ROOT/'artifacts/query-packing-v3/prefix-v2/queries')
   cmd+=['--hybrid-cache',str(packets if variant=='original' else ROOT/'artifacts/query-packing-v3/packets-v2'),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start']
   if variant=='short':cmd+=['--packed-start']
   cmd+=['--record',str(index),'--directory',str(dest)]
   with run_lock(dest):
    if not path.exists():run(cmd,d/f'{pair["id"]}-o{pair["offset"]}-{variant}.log')
    report=json.loads(path.read_text());row=extract(report,record,pair,variant,dest)
   rows.append(row)
   assert identities=={str(p.relative_to(ROOT)):sha(p) for p in paths},'sources changed during evaluation'
   current=dict(session,complete=False,completed_inferences=len(rows),elapsed_seconds_this_run=time.monotonic()-started,cases=rows,**summarize(fixture['pairs'],rows));atomic_json(d/'report.json',current)
   print(json.dumps(dict(completed=len(rows),total=64,id=pair['id'],offset=pair['offset'],variant=variant,gold=row['gold'],prediction=row['prediction'],correct=row['correct'],fallback=row['fallback'])),flush=True)
 current['complete']=True;atomic_json(d/'report.json',current);print(json.dumps(current['primary_distinct_cases']),flush=True)
if __name__=='__main__':main()
