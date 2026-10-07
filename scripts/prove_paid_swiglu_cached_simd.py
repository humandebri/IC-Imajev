#!/usr/bin/env python3
"""Prove paid SwiGLU inputs and billing; pause before cache-clearing upgrade tests."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-swiglu-cached-simd-v1';assert all(sha(ROOT/p)==h for p,h in json.loads((d/'workflow-hashes.json').read_text()).items());g=json.loads((d/'upgrade-guards/verified.json').read_text());b=json.loads((d/'build/report.json').read_text());assert g['complete'] and g['baseline_restored'] and g['module']==b['wasm_sha256']
 p=ROOT/'artifacts/paid-rms-ordered-simd-v1/frozen-proof.py';s=p.read_text().replace('artifacts/paid-rms-ordered-simd-v1','artifacts/paid-swiglu-cached-simd-v1')
 anchor='  # Use a tiny replacement module';i=s.index(anchor)
 pause="""  config['enabled']=False;config['version']=3;assert 'Ok'in wire.call('configure_paid',config)['result']
  paused_request=dict(request=requests[0],request_id='paused-before-upgrade',quote_version=3)
  paused_row=wire.call('infer',paused_request,relay=callers[0],cycles=quotes[0]['fee'])
  assert paused_row['result']=={'Err':'Paused'} and paused_row['forward']['refunded']==quotes[0]['fee']
  checks.append(dict(name='paused',row=paused_row));write(D/'checks.json',checks)
  config['enabled']=True;config['version']=4;assert 'Ok'in wire.call('configure_paid',config)['result']
  assert wire.call('quote',requests[0])['result']['Ok']['fee']==quotes[0]['fee']
"""
 s=s[:i]+pause+s[i:]
 before="active_request=dict(request=requests[0],request_id='upgrade-primary',quote_version=2)";assert s.count(before)==1;s=s.replace(before,before.replace('quote_version=2','quote_version=4'))
 lo=s.index("  # Same-version upgrade");tail=s[lo:].replace("config['version']=3","config['version']=5").replace("request_id='paused',quote_version=3","request_id='paused-after-concurrency',quote_version=5")
 before="assert row['result']=={'Err':'Paused'} and row['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='paused',row=row))";assert tail.count(before)==1
 tail=tail.replace(before,"assert row['result']=={'Err':('NotReady' if upgrade.returncode==0 else 'Paused')} and row['forward']['refunded']==quotes[0]['fee'];checks.append(dict(name='post-upgrade-readiness',row=row))")
 s=s[:lo]+tail;(d/'frozen-proof.py').write_text(s);files=[Path(__file__),p,d/'workflow-hashes.json',d/'guard-wrapper-hashes.json',d/'upgrade-guards/verified.json',d/'upgrade-guard-entry-hashes.json',d/'frozen-proof.py'];(d/'proof-entry-hashes.json').write_text(json.dumps({str(v.relative_to(ROOT)):sha(v)for v in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-proof.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
