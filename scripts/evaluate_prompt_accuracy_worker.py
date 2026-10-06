#!/usr/bin/env python3
"""Second inference lane; serialize shared journals with the serial controller."""
import json,pathlib,subprocess,sys,time
import evaluate_prompt_accuracy as e
R=e.ROOT;D=R/'artifacts/prompt-accuracy-v2';f=json.loads((D/'inputs.json').read_text());session=json.loads((D/'evaluation-session.json').read_text())
assert all(e.sha(R/k)==v for k,v in session['source_hashes'].items())
manifest=dict(scope='Independent read-only local query evaluations of pairs16..31 in reverse order; same prompts/module/flags as serial controller',controller_session_sha256=e.sha(D/'evaluation-session.json'),worker_sha256=e.sha(pathlib.Path(__file__)),completed=[])
e.atomic_json(D/'worker-report.json',manifest)
base=e.base_flags()
for pair in reversed(f['pairs'][16:]):
 for variant in ['original','short']:
  status=json.loads((D/'report.json').read_text())
  if status['completed_inferences']>=28:
   manifest['stopped_before_overlap']=True;e.atomic_json(D/'worker-report.json',manifest);sys.exit(0)
  index=pair[variant+'_record'];record=f['records'][index];dest=D/'runs'/f'{pair["id"]}-o{pair["offset"]}-{variant}';path=dest/'report.json'
  with e.run_lock(dest):
   if path.exists():continue
   assert all(e.sha(R/k)==v for k,v in session['source_hashes'].items())
   cmd=list(base);cmd[cmd.index('--cache')+1]=str(D/'original-prefix/queries' if variant=='original' else R/'artifacts/query-packing-v3/prefix-v2/queries')
   cmd+=['--hybrid-cache',str(D/'original-packets' if variant=='original' else R/'artifacts/query-packing-v3/packets-v2'),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start']
   if variant=='short':cmd+=['--packed-start']
   cmd+=['--record',str(index),'--directory',str(dest)]
   with (D/f'{pair["id"]}-o{pair["offset"]}-{variant}.log').open('a') as log:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
   row=e.extract(json.loads(path.read_text()),record,pair,variant,dest)
  manifest['completed'].append(row);e.atomic_json(D/'worker-report.json',manifest);print(json.dumps(dict(id=pair['id'],variant=variant,correct=row['correct'],completed=len(manifest['completed']))),flush=True)
manifest['complete_assigned_range']=True;e.atomic_json(D/'worker-report.json',manifest)
