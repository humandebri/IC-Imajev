#!/usr/bin/env python3
"""Independently decode both MLP paths and account for input-staging intervals."""
from pathlib import Path
import hashlib,json,subprocess,zipfile,argparse
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shape',choices=['gate','down'],required=True);a=ap.parse_args();d=ROOT/'artifacts/s2-integer-input-basis-probe-v1';c=d/('check-'+a.shape);r=json.loads((c/'report.json').read_text());b=json.loads((d/'build/report.json').read_text());normal=json.loads((ROOT/'artifacts/update-k2-pair-guard-fold-v1/build/report.json').read_text());k=ROOT/'artifacts/s2-cached-tail-dot-reuse-kernels-v1';kr=json.loads((k/'report.json').read_text());er=json.loads((k/'execution-report.json').read_text());identities={}
 for report in [r,b,normal,kr,er]:
  for key in ['source_hashes','dependency_hashes']:
   for f,h in report.get(key,{}).items():assert sha(ROOT/f)==h,f;identities[f]=h
 for name in ['builder-entry-hashes.json','checker-entry-hashes.json']:
  for f,h in json.loads((d/name).read_text()).items():assert sha(ROOT/f)==h,f;identities[f]=h
 assert r['wasm_sha256']==b['wasm_sha256']==sha(d/'build/diagnostic.wasm')
 assert len(b['patches'])==34 and [(v['export'],v['source_sha256'])for v in b['patches'][:30]]==[(v['export'],v['source_sha256'])for v in normal['patches']]
 for patch in b['patches'][30:]:assert any(patch['export']==v['symbol']and patch['source_sha256']==sha(ROOT/v['path'])for v in kr['kernels'])
 rows,cols=(9216,2560)if a.shape=='gate'else(2560,9216);assert len(r['cases'])==6 and r['ordinary_queries']==12
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';files=[Path(__file__),c/'report.json',d/'build/report.json',d/'builder-entry-hashes.json',d/'checker-entry-hashes.json'];replies=[];results=[]
 def decode(m,kind):
  p=ROOT/m['reply'];assert sha(p)==m['reply_sha256'];v=json.loads(subprocess.check_output([str(helper),'decode',str(p),kind],text=True));assert all(m[key]==value for key,value in v.items());replies.append(p);return v
 for m in r['preparations']:assert decode(m,'preparation')['instructions']<40_000_000_000
 for case in r['cases']:
  assert case['rows']==rows and case['cols']==cols and case['tokens']in [1,8,48,56,57,87]
  ip=c/(case['label']+'.input.bin');raw=ip.read_bytes();assert sha(ip)==case['input_sha256']and len(raw)==case['tokens']*cols*4;files.append(ip);parts=[];stage_cost=0
  assert len(case['input_staging'])==(len(raw)+999999)//1000000
  for i,m in enumerate(case['input_staging']):
   v=decode(m,'preparation');part=c/f'{case["label"]}-part-{i*1000000}.bin';data=part.read_bytes();assert data==raw[i*1000000:(i+1)*1000000]and v['bytes']==len(data);parts.append(data);files.append(part);stage_cost+=v['instructions']
  assert b''.join(parts)==raw
  for m in case['measurements'].values():
   v=decode(m,'measurement');assert v['digest']==case['native']['digest']and v['output_values']==rows*case['tokens'];assert v['total_instructions']==sum(v[key]for key in ['quantize_instructions','input_prepare_instructions','project_instructions'])
  before=case['measurements']['current_guard_fold_k2']['total_instructions'];after=case['measurements']['rank49_integer_input_basis']['total_instructions'];assert case['reduction_percent']==100*(1-after/before)
  results.append(dict(tokens=case['tokens'],rows=rows,cols=cols,before=before,after=after,input_staging_intervals=stage_cost,before_with_staging=before+stage_cost,after_with_staging=after+stage_cost,reduction_percent=case['reduction_percent'],native_bits_equal=True))
 files+=list(c.glob('*.args.bin'))+[ROOT/f for f in identities]+replies;files=list(dict.fromkeys(files));report=dict(complete=True,module=b['wasm_sha256'],shape=a.shape,cases=results,all_saved_Candid_redecoded=True,reply_count=len(replies),all_current_control_WAT_equal=True,input_staging_explicitly_counted=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},adopted=False,full_paid_goal_verified=False,scope='Synthetic-input MLP component intervals plus explicit input-staging intervals. Candid/digest and callbacks outside counter intervals are excluded. No full worker-handler or goal claim.')
 out=c/'summary.json';out.write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(c/'frozen-workflow.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[out]:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,shape=a.shape,cases=6,replies=len(replies),all_native_bits_equal=True)))
if __name__=='__main__':main()
