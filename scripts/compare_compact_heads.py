#!/usr/bin/env python3
"""Compare all layer outputs, retained state and decisions after wire compaction."""
import argparse,hashlib,json,pathlib,subprocess
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--before',required=True);ap.add_argument('--after',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 before=ROOT/args.before;after=ROOT/args.after
 old=json.loads((before/('first-report.json' if (before/'first-report.json').exists() else 'report.json')).read_text());new=json.loads((after/'report.json').read_text())
 for key in ['model','pack_hash','input_hash']:assert old[key]==new[key],key
 assert new['replayed_queries']==0
 assert len(new['layers'])==32
 def equal(a,b):assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
 terminal=bool(new.get("terminal_readout"));states=0;omitted=0;log_verified=0;baseline_log_verified=0;verifier_hash=None
 def restore_log(values,directory,layer):
  nonlocal verifier_hash
  log=values.pop('delta_log');assert log.ndim==1 and log.size%6176==0
  directory.mkdir(exist_ok=True);inp=directory/f'delta-log-{layer:02d}.bin';out=directory/f'delta-state-{layer:02d}.bin';inp.write_bytes(log.astype('<f4').tobytes())
  helper=ROOT/'target/release/delta_log_verify';verifier_hash=hashlib.sha256(helper.read_bytes()).hexdigest()
  subprocess.run([str(helper),str(log.size//6176),str(inp),str(out)],check=True,cwd=ROOT)
  values['delta']=np.frombuffer(out.read_bytes(),'<f4').reshape(32,128,128)
 for layer in range(32):
  a=np.load(before/f'queries/layer-{layer:02d}.npy');b=np.load(after/f'queries/layer-{layer:02d}.npy')
  if terminal and layer==31:
   assert b.shape==(1,a.shape[1]);a=a[-1:]
  equal(a,b)
  with np.load(before/f'queries/states/layer-{layer:02d}.npz') as a,np.load(after/f'queries/states/layer-{layer:02d}.npz') as b:
   a={k:a[k].copy() for k in a.files};b={k:b[k].copy() for k in b.files}
   if 'delta_log' in a and 'delta_log' in b:equal(a['delta_log'],b['delta_log'])
   if 'delta_log' in a:
    restore_log(a,after/'verification-baseline',layer);baseline_log_verified+=1
   if 'delta_log' in b:
    restore_log(b,after/'verification',layer);log_verified+=1
   assert set(b)<=set(a)
   missing=set(a)-set(b)
   assert not missing or (missing=={'delta'} and (layer+1)%4!=0)
   omitted+=len(missing)
   for key in b:equal(a[key],b[key]);states+=1
 if 'decision_query' in new:
  equal(np.load(before/'final-hidden.npy')[-1:] if terminal else np.load(before/'final-hidden.npy'),np.load(after/'final-hidden.npy'))
  a=old['decision_query']['ok']['decision'];b=new['decision_query']['ok']['decision']
  for key in ['raw_logits','probabilities','unknown_probability','value','abstained']:assert a[key]==b[key],key
  assert new['comparison']['typed_output_valid']
 metrics=['query_count','total_instructions','total_candid_bytes','wall_seconds_this_run','max_query_instructions','max_observed_heap_bytes']
 report=dict(before={k:old[k] for k in metrics},after={k:new[k] for k in metrics},all_32_hidden_bitwise_equal=not terminal,all_retained_hidden_bitwise_equal=True,full_hidden_layers=31 if terminal else 32,final_layer_last_token_only=terminal,retained_state_arrays_bitwise_equal=states,terminal_state_arrays_omitted=omitted,offline_delta_log_state_verifications=log_verified,offline_delta_log_verifier_sha256=verifier_hash,decision_identical=True if 'decision_query' in new else None,wasm_sha256=new['wasm_sha256'],comparison=new.get('comparison'),queries=new['queries'],raw_report_sha256=hashlib.sha256((after/'report.json').read_bytes()).hexdigest())
 report['offline_baseline_delta_log_state_verifications']=baseline_log_verified
 (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='queries'},indent=2))
if __name__=='__main__':main()
