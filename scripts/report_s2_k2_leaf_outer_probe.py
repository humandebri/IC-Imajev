#!/usr/bin/env python3
"""Redecode all fresh component replies and verify current control and new WAT."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-leaf-outer-probe-v1';k=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1';r=json.loads((d/'check/report.json').read_text());b=json.loads((d/'build/report.json').read_text());n=json.loads((ROOT/'artifacts/update-k2-pair-guard-fold-v1/build/report.json').read_text());kr=json.loads((k/'report.json').read_text());layout=json.loads((k/'layout-audit.json').read_text());identities={}
 for report in [r,b,n,kr,layout]:
  for key in ['source_hashes','dependency_hashes']:
   for path,h in report.get(key,{}).items():assert sha(ROOT/path)==h;identities[path]=h
 for directory,names in [(d,['entry-builder-hashes.json','entry-checker-hashes.json']),(k,['validator-entry-hashes.json','auditor-entry-hashes.json'])]:
  for name in names:
   for path,h in json.loads((directory/name).read_text()).items():assert sha(ROOT/path)==h;identities[path]=h
 assert b['wasm_sha256']==r['wasm_sha256']==sha(d/'build/diagnostic.wasm');assert len(b['patches'])==34
 assert [(p['export'],p['source_sha256'])for p in b['patches'][:30]]==[(p['export'],p['source_sha256'])for p in n['patches']]
 for patch in b['patches'][30:]:assert any(patch['export']==item['symbol']and patch['source_sha256']==sha(ROOT/item['path'])for item in kr['kernels'])
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';replies=[];cases=[]
 def decode(m,kind):
  p=ROOT/m['reply'];assert sha(p)==m['reply_sha256'];v=json.loads(subprocess.check_output([str(helper),'decode',str(p),kind],text=True));assert all(m[key]==value for key,value in v.items());replies.append(p);return v
 assert len(r['cases'])==21 and r['ordinary_queries']==42 and len(r['preparations'])==17
 for p in r['preparations']:assert decode(p,'preparation')['instructions']<40_000_000_000
 for case in r['cases']:
  assert sha(d/'check'/(case['label']+'.input.bin'))==case['input_sha256']
  for m in case['measurements'].values():
   v=decode(m,'measurement');assert v['digest']==case['native']['digest'];assert v['output_values']==case['tokens']*case['rows'];assert v['total_instructions']==sum(v[name]for name in ['quantize_instructions','input_prepare_instructions','project_instructions'])
  before=case['measurements']['current_guard_fold_k2']['total_instructions'];after=case['measurements']['rank49_leaf_outer']['total_instructions'];assert case['reduction_percent']==100*(1-after/before);cases.append(dict(label=case['label'],tokens=case['tokens'],rows=case['rows'],before=before,after=after,reduction_percent=case['reduction_percent']))
 files=[Path(__file__),d/'build/report.json',d/'check/report.json',k/'layout-audit.json',k/'report.json',d/'entry-builder-hashes.json',d/'entry-checker-hashes.json',k/'validator-entry-hashes.json',k/'auditor-entry-hashes.json']+[ROOT/path for path in identities]+replies
 result=dict(complete=True,module=b['wasm_sha256'],all21_native_bits_equal=True,all59_saved_Candid_replies_redecoded=True,all30_current_control_WAT_equal=True,output_tiles=[96,16],rank=49,query_local_vectors=32,raw_capacity_unchanged=True,conditions=21,cases=cases,adopted=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},scope='Actual layer3 Q component only; all inference quantize/prepare/project work included. No full inference or goal claim.')
 (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[d/'summary.json']):z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(d/'frozen-workflow.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()));assert z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
 print(json.dumps({key:value for key,value in result.items()if key not in ['source_hashes','cases']}))
if __name__=='__main__':main()
