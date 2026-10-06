#!/usr/bin/env python3
"""Run disjoint lanes of the frozen subset on the existing local query canister."""
import argparse, concurrent.futures, json, pathlib, statistics, subprocess, time
import evaluate_prompt_accuracy as e
R = e.ROOT
D = R / 'artifacts/decision-index-v1'

def summarize():
    fixture = json.loads((D/'inputs.json').read_text())
    rows, errors = [], []
    for i, record in enumerate(fixture['records']):
        directory = D/'runs'/f'{i:03d}'
        path = directory/'report.json'
        if not path.exists():
            if (directory/'error.json').exists():
                errors.append(json.loads((directory/'error.json').read_text()))
            continue
        report = json.loads(path.read_text())
        decision = report['decision_query']['ok']['decision']
        assert report['wasm_sha256'] == e.MODULE
        assert report['input_hash'] == record['input_sha256']
        assert report['comparison']['typed_output_valid'] and report['comparison']['gold'] == record['gold']
        prediction = '__unknown__' if decision['abstained'] else decision['value']
        rows.append(dict(index=i,dataset=record['dataset'],source_id=record['source_id'],gold=record['gold'],prediction=prediction,
                         correct=prediction==record['gold'],tokens=report['tokens'],query_count=report['query_count'],
                         total_instructions=report['total_instructions'],total_candid_bytes=report['total_candid_bytes'],
                         measured_successful_instructions=report.get('successful_query_instructions',report['total_instructions']),
                         measured_successful_candid_bytes=report.get('successful_query_candid_bytes',report['total_candid_bytes']),
                         seconds=report.get('end_to_end_seconds_excluding_process_startup',report['wall_seconds_this_run']),replayed_queries=report['replayed_queries'],
                         fallback=bool(report.get('fallback')), probabilities=decision['probabilities'],unknown_probability=decision['unknown_probability'],
                         report=str(path.relative_to(R)),report_sha256=e.sha(path)))
    grouped={}
    for family in fixture['inventory']:
        group=[r for r in rows if r['dataset']==family]
        grouped[family]=dict(planned=20,completed=len(group),correct=sum(r['correct'] for r in group),
                            accuracy=sum(r['correct'] for r in group)/20,
                            unknown=sum(r['prediction']=='__unknown__' for r in group))
    result=dict(scope=fixture['scope'],inputs_sha256=e.sha(D/'inputs.json'),module=e.MODULE,planned=100,completed=len(rows),
                complete=len(rows)+len(errors)==100,errors=errors,correct=sum(r['correct'] for r in rows),
                accuracy=sum(r['correct'] for r in rows)/100,unknown=sum(r['prediction']=='__unknown__' for r in rows),
                by_dataset=grouped,cases=rows)
    if rows:
        result['execution']=dict(total_queries=sum(r['query_count'] for r in rows),total_candid_bytes=sum(r['total_candid_bytes'] for r in rows) if all(r['total_candid_bytes'] is not None for r in rows) else None,
                                 median_queries=statistics.median(r['query_count'] for r in rows),min_queries=min(r['query_count'] for r in rows),max_queries=max(r['query_count'] for r in rows),
                                 median_successful_candid_bytes=statistics.median(r['measured_successful_candid_bytes'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows),
                                 total_instructions=sum(r['total_instructions'] for r in rows) if all(r['total_instructions'] is not None for r in rows) else None,replayed_queries=sum(r['replayed_queries'] for r in rows),fallbacks=sum(r['fallback'] for r in rows))
    e.atomic_json(D/'report.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--summarize',action='store_true');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.summarize:
        result=summarize();print(json.dumps({k:v for k,v in result.items() if k!='cases'}));return
    fixture=json.loads((D/'inputs.json').read_text())
    paths=[D/'inputs.json',R/'MODEL_LOCK.json',R/'artifacts/query-packing-v3/build/full.wasm',R/'artifacts/query-packing-v3/build/imajev-client',R/'scripts/run_prefix_canister.py',R/'client/full_inference.py',R/'client/prefix_inference.py',D/'dense-prefix/queries/cache.json',pathlib.Path(__file__)]
    identities={str(p.relative_to(R)):e.sha(p) for p in paths}
    session=dict(input_hash=e.sha(D/'inputs.json'),source_hashes=identities,workers=args.workers,planned=100,module=e.MODULE)
    path=D/'large-evaluation-session.json'
    if path.exists():assert json.loads(path.read_text())==session
    else:e.atomic_json(path,session)
    def run(i):
        parent=D/'runs'/f'{i:03d}';dest=parent/'large';dest.mkdir(parents=True,exist_ok=True)
        if (parent/'report.json').exists():return i
        assert all(e.sha(R/k)==v for k,v in identities.items()),'Evaluation sources changed'
        cmd=e.base_flags();cmd[cmd.index('--reference')+1]=str(D/'inputs.json');cmd[cmd.index('--cache')+1]=str(D/'dense-prefix/queries')
        for flag in ['--fuse-delta-full-log','--fuse-delta-projected','--fuse-delta-finish','--fuse-attention','--fuse-attention-full']:
            cmd.remove(flag)
        cmd+=['--terminal-readout','--record',str(i),'--directory',str(dest)]
        with (dest/'run.log').open('a') as log:
            process=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            e.atomic_json(dest/'error.json',dict(index=i,source_id=fixture['records'][i]['source_id'],returncode=process.returncode,log=str((dest/'run.log').relative_to(R))))
        else:
            report=json.loads((dest/'report.json').read_text());report['successful_run_directory']=str(dest)
            e.atomic_json(parent/'report.json',report)
        assert all(e.sha(R/k)==v for k,v in identities.items()),'Evaluation sources changed'
        return i
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,i) for i,r in enumerate(fixture['records']) if len(r['token_ids'])-26>132]
        for future in concurrent.futures.as_completed(futures):
            index=future.result();result=summarize()
            print(json.dumps(dict(index=index,completed=result['completed'],errors=len(result['errors']),correct=result['correct'])),flush=True)
    result=summarize();print(json.dumps(result['by_dataset']),flush=True)

if __name__=='__main__':main()
