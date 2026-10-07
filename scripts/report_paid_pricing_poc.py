#!/usr/bin/env python3
"""Re-decode payment evidence and summarize approximate POC pricing."""
import json,subprocess,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/(sys.argv[1] if len(sys.argv)>1 else 'artifacts/paid-pricing-poc-v1')
def load(p):return json.loads(p.read_text())
def main():
 report=load(D/'report.json');assert report['complete'];config=report['config'];rows={};helper=R/'artifacts/paid-update-v1/tools/args'
 for path in sorted((D/'calls').glob('[0-9][0-9][0-9][0-9]-*.json')):
  row=load(path)
  if not isinstance(row,dict) or 'kind' not in row:continue
  raw=R/row['reply_path'];kind='forward' if row['relay'] else row['kind'];decoded=json.loads(subprocess.check_output([str(helper),'decode',kind,str(raw)],text=True))
  if row['relay']:
   assert decoded==row['forward'];assert 'Ok' in decoded['response'];inner=raw.with_name(raw.name.replace('.reply.hex','.inner.hex'));assert inner.read_text().strip()==bytes(decoded['response']['Ok']).hex()
   decoded=json.loads(subprocess.check_output([str(helper),'decode',row['kind'],str(inner)],text=True))
  assert decoded==row['result'];rows[row['reply_path']]=row
 assert len(report['results'])==(4 if D.name.endswith('v2') else 9)
 summary=[]
 for case in report['results']:
  row=rows[case['row']['reply_path']];assert row==case['row'];result=row['result']['Ok'];q=case['quote'];n=q['suffix_tokens'];workers=result['workers']
  assert result['paid_cycles']==q['fee'] and row['forward']['refunded']==12345678
  assert len(workers)<=6 and all(w['instructions']<40000000000 and w['heap_pages']*65536<2**32 for w in workers)
  instructions=sum(w['instructions'] for w in workers);assert instructions==case['instructions']
  # 13-node cost estimate: worker instructions + 1B for uncounted operations.
  estimated_variable=instructions+1000000000;newfee=config['base_fee']+config['fee_per_token']*n
  assert newfee>=instructions*1.2+1000000000
  if q['version']==3:assert q['fee']==newfee
  else:assert q['fee']==100000000000+3000000000*n
  before=case['before'];after=case['after'];burn=before['cycles']+before['reserved']+q['fee']-after['cycles']-after['reserved'];assert burn==case['balance_burn'];assert newfee>=burn*1.2+1000000000
  if case['name'].startswith('new-fee-'):
   name=case['name'].removeprefix('new-fee-');ref=load(R/f'artifacts/boomdao-current-v1/{name}-r1/report.json')['decision_query']['ok']['decision']
   assert {k:v for k,v in result['decision'].items() if k!='instructions'}=={k:v for k,v in ref.items() if k!='instructions'}
  summary.append(dict(name=case['name'],suffix=n,prefix=q['prefix_tokens'],workers=len(workers),instructions=instructions,local_balance_burn=burn,local_variable_estimate=case['variable_estimate'],estimated_13_node_variable=estimated_variable,poc_fee=newfee,old_fee=100000000000+3000000000*n,seconds=row['seconds']))
 checks={c['name']:rows[c['row']['reply_path']] for c in report['checks']};assert set(checks)=={'duplicate-new-fee','insufficient','old-version'}
 assert checks['duplicate-new-fee']['forward']['refunded']==checks['duplicate-new-fee']['attached_cycles']
 assert 'InsufficientCycles' in checks['insufficient']['result']['Err'] and checks['insufficient']['forward']['refunded']==1
 assert 'QuoteChanged' in checks['old-version']['result']['Err'] and checks['old-version']['forward']['refunded']==checks['old-version']['attached_cycles']
 assert load(D/'restored.json')==dict(module='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931',cache_equal=True,pack_equal=True,snapshot_deleted=True)
 assert hashlib.sha256((R/'artifacts/paid-update-v1/build-v3/full.wasm').read_bytes()).hexdigest()==report['candidate']
 if D.name.endswith('v2'):
  calibration=load(R/'artifacts/paid-pricing-poc-v1/progress.json')[:6]
  assert config['base_fee']==10000000000 and config['fee_per_token']==3000000000
  assert all(config['base_fee']+config['fee_per_token']*r['quote']['suffix_tokens']>=max(r['instructions'],r['balance_burn'])*1.2+1000000000 for r in calibration)
 result=dict(complete=True,scope='POC approximate tariff; local balance burn not claimed as 13-node mainnet cost. Storage subsidized.',config=config,cases=summary,raw_responses_verified=len(rows),baseline_restored=True)
 (D/'verified.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
