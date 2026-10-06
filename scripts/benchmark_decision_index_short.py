#!/usr/bin/env python3
"""Evaluate the frozen short-input sample using real local canister queries."""
import argparse,concurrent.futures,collections,json,pathlib,statistics,subprocess
import evaluate_prompt_accuracy as e
R=e.ROOT;D=R/'artifacts/decision-index-short-v1'

def caches(prefix):
    base=R/('artifacts/query-packing-v3/prefix-v2' if prefix==27 else 'artifacts/decision-index-v1/prefix')
    packets=R/('artifacts/query-packing-v3/packets-v2' if prefix==27 else 'artifacts/decision-index-v1/packets')
    return base/'queries',packets

def summarize():
    fixture=json.loads((D/'inputs.json').read_text());rows=[];errors=[]
    for i,record in enumerate(fixture['records']):
        directory=D/'runs'/f'{i:03d}';path=directory/'report.json'
        if not path.exists():
            if (directory/'error.json').exists():errors.append(json.loads((directory/'error.json').read_text()))
            continue
        report=json.loads(path.read_text());decision=report['decision_query']['ok']['decision']
        assert report['wasm_sha256']==e.MODULE and report['input_hash']==record['input_sha256']
        assert report['model']==fixture['model_lock_sha256'] and report['comparison']['typed_output_valid']
        assert report['comparison']['gold']==record['gold']
        prediction='__unknown__' if decision['abstained'] else decision['value']
        rows.append(dict(index=i,dataset=record['dataset'],source_id=record['source_id'],gold=record['gold'],prediction=prediction,
                         correct=prediction==record['gold'],options=record['options'],tokens=report['tokens'],prefix_tokens=record['prefix_tokens'],
                         query_count=report['query_count'],candid_bytes=report['total_candid_bytes'],instructions=report['total_instructions'],
                         seconds=report['end_to_end_seconds_excluding_process_startup'],replayed_queries=report['replayed_queries'],fallback=bool(report.get('fallback')),
                         probabilities=decision['probabilities'],unknown_probability=decision['unknown_probability'],report=str(path.relative_to(R)),report_sha256=e.sha(path)))
    grouped={}
    for family in fixture['inventory']:
        group=[r for r in rows if r['dataset']==family]
        grouped[family]=dict(planned=20,completed=len(group),correct=sum(r['correct'] for r in group),accuracy=sum(r['correct'] for r in group)/len(group) if group else None,
                            gold_counts=dict(collections.Counter(r['gold'] for r in group)),prediction_counts=dict(collections.Counter(r['prediction'] for r in group)))
    result=dict(scope=fixture['scope'],inputs_sha256=e.sha(D/'inputs.json'),planned=100,completed=len(rows),complete=len(rows)==100 and not errors,
                correct=sum(r['correct'] for r in rows),accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,
                unknown=sum(r['prediction']=='__unknown__' for r in rows),errors=errors,by_dataset=grouped,cases=rows)
    if rows:
        result['execution']=dict(total_queries=sum(r['query_count'] for r in rows),median_queries=statistics.median(r['query_count'] for r in rows),
                                 min_queries=min(r['query_count'] for r in rows),max_queries=max(r['query_count'] for r in rows),
                                 total_candid_bytes=sum(r['candid_bytes'] for r in rows),median_candid_bytes=statistics.median(r['candid_bytes'] for r in rows),
                                 total_instructions=sum(r['instructions'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows),
                                 replayed_queries=sum(r['replayed_queries'] for r in rows),fallbacks=sum(r['fallback'] for r in rows))
    e.atomic_json(D/'report.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,choices=[1,2],default=2);parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    if args.summarize:
        print(json.dumps({k:v for k,v in summarize().items() if k!='cases'}));return
    fixture=json.loads((D/'inputs.json').read_text());assert len(fixture['records'])==100
    paths=[D/'inputs.json',R/'MODEL_LOCK.json',R/'artifacts/query-packing-v3/build/full.wasm',R/'artifacts/query-packing-v3/build/imajev-client',
           R/'scripts/run_prefix_canister.py',R/'scripts/evaluate_prompt_accuracy.py',pathlib.Path(__file__)]
    paths+=sorted((R/'client').glob('*.py'))
    paths+=[path/'cache.json' for prefix in [26,27] for path in caches(prefix)]
    identities={str(path.relative_to(R)):e.sha(path) for path in paths}
    session=dict(planned=100,module=e.MODULE,input_hash=e.sha(D/'inputs.json'),source_hashes=identities,workers=args.workers)
    path=D/'evaluation-session.json'
    if path.exists():assert json.loads(path.read_text())==session
    else:e.atomic_json(path,session)
    def run(i):
        record=fixture['records'][i];assert 1<=len(record['token_ids'])-record['prefix_tokens']<=87
        directory=D/'runs'/f'{i:03d}';directory.mkdir(parents=True,exist_ok=True)
        if (directory/'report.json').exists():return
        assert all(e.sha(R/p)==h for p,h in identities.items()),'Sources changed'
        cache,packets=caches(record['prefix_tokens'])
        cmd=e.base_flags();cmd[cmd.index('--reference')+1]=str(D/'inputs.json');cmd[cmd.index('--cache')+1]=str(cache)
        cmd+=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision',
              '--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--packed-start','--record',str(i),'--directory',str(directory)]
        with (directory/'run.log').open('a') as log:process=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:e.atomic_json(directory/'error.json',dict(index=i,returncode=process.returncode,log=str((directory/'run.log').relative_to(R))))
        assert all(e.sha(R/p)==h for p,h in identities.items()),'Sources changed'
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(run,i) for i in range(100)]
        for task in concurrent.futures.as_completed(tasks):
            task.result();result=summarize();print(json.dumps(dict(completed=result['completed'],correct=result['correct'],errors=len(result['errors']))),flush=True)
    result=summarize();assert result['complete'],'Incomplete evaluation; inspect errors'

if __name__=='__main__':main()
