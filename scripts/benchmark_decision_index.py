#!/usr/bin/env python3
"""Repeat the frozen 100-question canister benchmark with length-aware routing."""
import argparse,concurrent.futures,json,pathlib,subprocess
import evaluate_prompt_accuracy as e
from evaluate_decision_index_subset import summarize
R=e.ROOT;D=R/'artifacts/decision-index-v1'

def command(i,record):
    n=len(record['token_ids'])-26
    parent=D/'runs'/f'{i:03d}'
    directory=parent if n<=132 else parent/'large'
    cmd=e.base_flags()
    cmd[cmd.index('--reference')+1]=str(D/'inputs.json')
    cmd[cmd.index('--cache')+1]=str(D/('prefix/queries' if n<=90 else 'dense-prefix/queries'))
    if n<=90:
        cmd+=['--hybrid-cache',str(D/'packets'),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision',
              '--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--packed-start']
    else:
        cmd.remove('--fuse-delta-full-log')
        if n<=132:
            cmd+=['--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision']
        else:
            for flag in ['--fuse-delta-projected','--fuse-delta-finish','--fuse-attention','--fuse-attention-full']:
                cmd.remove(flag)
            cmd+=['--terminal-readout']
    cmd+=['--record',str(i),'--directory',str(directory)]
    return cmd,parent,directory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,choices=[1,2],default=1);args=parser.parse_args()
    fixture=json.loads((D/'inputs.json').read_text())
    assert len(fixture['records'])==100
    paths=[D/'inputs.json',R/'MODEL_LOCK.json',R/'scripts/run_prefix_canister.py',R/'scripts/evaluate_prompt_accuracy.py',
           R/'artifacts/query-packing-v3/build/full.wasm',R/'artifacts/query-packing-v3/build/imajev-client',
           D/'prefix/queries/cache.json',D/'dense-prefix/queries/cache.json',D/'packets/cache.json',pathlib.Path(__file__)]
    paths+=sorted((R/'client').glob('*.py'))
    hashes={str(p.relative_to(R)):e.sha(p) for p in paths}
    session=dict(planned=100,module=e.MODULE,input_hash=e.sha(D/'inputs.json'),source_hashes=hashes,workers=args.workers)
    target=D/'benchmark-session.json'
    if target.exists():assert json.loads(target.read_text())==session
    else:e.atomic_json(target,session)
    def execute(i,record):
        cmd,parent,directory=command(i,record)
        if (parent/'report.json').exists():return
        assert all(e.sha(R/p)==h for p,h in hashes.items()),'Sources changed'
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'run.log').open('a') as log:
            result=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            e.atomic_json(parent/'error.json',dict(index=i,source_id=record['source_id'],returncode=result.returncode,
                                                   log=str((directory/'run.log').relative_to(R))))
            return
        if directory!=parent:
            report=json.loads((directory/'report.json').read_text());report['successful_run_directory']=str(directory)
            e.atomic_json(parent/'report.json',report)
        assert all(e.sha(R/p)==h for p,h in hashes.items()),'Sources changed'
    pending=[(i,r) for i,r in enumerate(fixture['records']) if not (D/'runs'/f'{i:03d}'/'report.json').exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(execute,i,r) for i,r in pending]
        for task in concurrent.futures.as_completed(tasks):
            task.result();result=summarize()
            print(json.dumps(dict(completed=result['completed'],planned=100,correct=result['correct'])),flush=True)
    result=summarize()
    assert result['completed']==100 and not result['errors'],'Incomplete evaluation; inspect recorded errors'
    subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/verify_decision_index_subset.py')],cwd=R,check=True)
    subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/report_decision_index_subset.py')],cwd=R,check=True)

if __name__=='__main__':main()
