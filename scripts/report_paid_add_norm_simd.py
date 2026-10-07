#!/usr/bin/env python3
"""Audit raw paid Candid, worker counters, exact references, billing and restoration."""
from pathlib import Path
import argparse,hashlib,json,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(np.asarray(v,dtype='<f4').tobytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 d=ROOT/'artifacts/paid-add-norm-simd-v2';p=d/'proof';b=json.loads((d/'build/report.json').read_text())
 if args.partial:
  r=dict(candidate=b['wasm_sha256'],results=json.loads((p/'progress.json').read_text()),checks=json.loads((p/'checks.json').read_text()))
 else:
  r=json.loads((p/('progress.json' if args.partial else 'report.json')).read_text())
 if not args.partial:assert r['complete'] and r['baseline_restored'] and r['snapshot_deleted'] and r['dual_bank'] and r['upgrade_receipts_equal']
 assert sha(d/'build/full.wasm')==r['candidate']==b['wasm_sha256']
 manifests=[json.loads((d/n).read_text()) for n in ['workflow-hashes.json','proof-entry-hashes.json']]+[b['source_hashes'],b['dependency_hashes'],json.loads((p/'sources.json').read_text())]
 for m in manifests:assert all(sha(ROOT/v)==h for v,h in m.items())
 assert b['canonical_billing_sources_byte_equal'] and b['bound_prefix_and_bulk_digest']
 for n in ['paid_types.rs','paid_inference.rs']:assert (d/'build'/n).read_bytes()==(ROOT/'canisters/inference/src'/n).read_bytes()
 assert (d/'build/update_inference.rs').read_bytes()==(d/'optimized-paid-scheduler.rs').read_bytes()
 restored=json.loads((p/'restored.json').read_text());before=json.loads((p/'before.json').read_text());assert restored['module']==before['module']=='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931';assert restored['cache_equal'] and restored['pack_equal'] and restored['snapshot_deleted']
 helper=ROOT/'artifacts/paid-update-v1/tools/args';calls=[];paid_results=[];debugs=[]
 def decode(kind,file):return json.loads(subprocess.check_output([str(helper),'decode',kind,str(file)],text=True))
 for v in sorted(p.rglob('*.json')):
  c=json.loads(v.read_text())
  if not isinstance(c,dict) or not {'kind','reply_path','args_path','result'}<=c.keys():continue
  reply=ROOT/c['reply_path'];argument_file=ROOT/c['args_path'];assert argument_file.stat().st_size==c['request_bytes'] and len(reply.read_text().strip().removeprefix('0x'))//2==c['reply_bytes']
  if c['relay']:
   f=decode('forward',reply);assert f==c['forward'];response=f['response']
   if 'Ok' in response:
    inner=reply.with_name(reply.name.replace('.reply.hex','.inner.hex'));assert bytes.fromhex(inner.read_text().strip().removeprefix('0x'))==bytes(response['Ok']);actual=decode(c['kind'],inner)
   else:actual=dict(transport_error=response['Err'])
  else:actual=decode(c['kind'],reply)
  assert actual==c['result'],str(v)
  if c['kind']=='infer' and 'Ok' in actual:paid_results.append(actual['Ok'])
  if c['kind']=='paid_debug':debugs.append(actual)
  calls.append(v)
 cases=[];references=[]
 assert len(r['results'])==3
 for item in r['results']:
  name=item['case'];result=item['row']['result']['Ok'];debug=item['debug'];assert result in paid_results and debug in debugs and debug['job_id']==result['job_id']
  assert item['row']['forward']['refunded']==12_345_678 and result['paid_cycles']==item['quote']['fee']
  assert all(v['instructions']<40_000_000_000 and v['heap_pages']*65536<2**32 for v in result['workers'])
  old=ROOT/f'artifacts/boomdao-current-v1/{name}-r1';refpath=old/'report.json';reference=json.loads(refpath.read_text())['decision_query']['ok']['decision'];actual=result['decision'];assert {k:v for k,v in actual.items() if k!='instructions'}=={k:v for k,v in reference.items() if k!='instructions'}
  for k in ['raw_logits','probabilities']:assert digest(actual[k])==digest(reference[k])
  assert digest([actual['unknown_probability']])==digest([reference['unknown_probability']]);references.append(refpath)
  final=old/'final-hidden.npy';assert digest(debug['final_hidden'])==digest(np.load(final,allow_pickle=False));references.append(final);prefix=item['quote']['prefix_tokens'];assert prefix==(38 if name in ['617','620'] else 27)
  for layer in range(32):
   hidden=old/f'queries/layer-{layer:02d}.npy'
   if hidden.exists():
    a=np.load(hidden,allow_pickle=False);a=a[prefix:] if layer<31 else a;assert digest(a)==debug['hidden_hashes'][layer];references.append(hidden)
   else:assert layer==30
   state=old/f'queries/states/layer-{layer:02d}.npz'
   with np.load(state,allow_pickle=False)as z:a=np.concatenate([z['keys'][prefix:].ravel(),z['values'][prefix:].ravel()]) if layer%4==3 else z['conv'].ravel();assert digest(a)==debug['state_hashes'][layer]
   references.append(state)
  total=sum(v['instructions'] for v in result['workers']);cases.append(dict(case=name,total_handler_instructions=total,workers=len(result['workers']),max_heap_bytes=max(v['heap_pages']*65536 for v in result['workers']),target_met=total<=100_000_000_000))
 assert {v['case'] for v in cases}=={'617','620','653'}
 if not args.partial:
  concurrency=json.loads((p/'concurrency.json').read_text());assert concurrency['active_upgrade_refused'] and concurrency['busy']['result']=={'Err':'Busy'} and 'Ok' in concurrency['primary']['result']
 check_names={v['name'] for v in r['checks']};required={'duplicate','id-conflict','insufficient','quote-version','token-bounds','worker-authority','status-authority'}
 if not args.partial:required|={'paused','upgrade-replay'}
 assert required<=check_names
 files=[Path(__file__),d/'workflow-hashes.json',d/'proof-entry-hashes.json',d/'frozen-proof.py',d/'build/report.json',p/('progress.json' if args.partial else 'report.json'),p/'sources.json',p/'before.json',p/'restored.json',p/('active-upgrade.json' if args.partial else 'concurrency.json'),p/'checks.json']+calls
 summary=dict(module=r['candidate'],paid_core_api_verified=True,complete=not args.partial,saved_candid_replies_verified=True,baseline_restored=True,payment_boundary_checks_verified=True,upgrade_checks_verified=not args.partial,cases=cases,all_targets_met=all(v['target_met'] for v in cases),reference_hashes={str(v.relative_to(ROOT)):sha(v) for v in references},workflow_hashes={str(v.relative_to(ROOT)):sha(v) for v in files},scope='Actual caller-canister paid API on three inputs. 31 exported hidden, 32 conv/KV state hashes, final norm and decision/probability F32 bits; layer30 hidden and dense Delta state not directly exported. Full worker totals include original start and scheduler work.')
 summary_name='partial-summary.json' if args.partial else 'summary.json'
 if args.partial:summary['failed_extra_test']='Concurrent active upgrade succeeded; execution-order race unresolved. Paused and same-version upgrade checks not executed.'
 (d/summary_name).write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(d/('frozen-paid-partial-proof.zip' if args.partial else 'frozen-paid-proof.zip'),'w',zipfile.ZIP_DEFLATED)as z:
  for v in files+list(p.rglob('*.hex'))+list(p.rglob('*.args.bin'))+[d/summary_name]:z.write(v,str(v.relative_to(ROOT)))
 print(json.dumps(cases,indent=2))
if __name__=='__main__':main()
