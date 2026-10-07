#!/usr/bin/env python3
"""Redecode saved replies, current control hashes and complete timing arithmetic."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/rank7-batched256-probe-v5';check=d/'check';r=json.loads((check/'report.json').read_text());b=json.loads((d/'build/report.json').read_text());normal=json.loads((ROOT/'artifacts/update-common-raw-hybrid-v1/build/report.json').read_text())
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';identities={}
 for report in [r,b,normal,json.loads((ROOT/'artifacts/rank7-batched256-v4/report.json').read_text()),json.loads((ROOT/'artifacts/rank7-batched256-v4/execution-report.json').read_text())]:
  groups=[report[key]for key in ['source_hashes','dependency_hashes']if key in report]
  if not groups:groups=[report]
  for group in groups:
   for path,digest in group.items():assert sha(ROOT/path)==digest,(path,digest);assert path not in identities or identities[path]==digest;identities[path]=digest
 assert b['wasm_sha256']==r['wasm_sha256']==sha(d/'build/diagnostic.wasm')
 assert len(b['patches'])==36 and len(normal['patches'])==34
 assert [(p['export'],p['source_sha256'])for p in b['patches'][:34]]==[(p['export'],p['source_sha256'])for p in normal['patches']]
 replies=[]
 def decode(measurement,kind):
  p=ROOT/measurement['reply'];assert sha(p)==measurement['reply_sha256'];value=json.loads(subprocess.check_output([str(helper),'decode',str(p),kind],text=True));assert all(measurement[k]==v for k,v in value.items());replies.append(p);return value
 assert len(r['preparations'])==17
 for prep in r['preparations']:assert decode(prep,'preparation')['instructions']<40_000_000_000

 assert len(r['cases'])==21 and r['ordinary_queries']==42
 changes=[]
 for case in r['cases']:
  assert sha(check/(case['label']+'.input.bin'))==case['input_sha256']
  ip=check/(case['label']+'.input.bin');wp=check/f"weights-{case['rows']}-native.bin"
  native=json.loads(subprocess.check_output([str(helper),'native',str(case['tokens']),str(case['rows']),'2560',str(wp),str(ip)],text=True));assert native==case['native']
  measures=case['measurements']
  for key,m in measures.items():
   v=decode(m,'measurement');assert v['digest']==case['native']['digest'];assert v['total_instructions']==v['quantize_instructions']+v['input_prepare_instructions']+v['project_instructions'];assert v['output_values']==case['rows']*case['tokens']
  before=measures['current_common_raw_hybrid']['total_instructions'];after=measures['rank7_batched256']['total_instructions'];reduction=100*(1-after/before);assert reduction==case['reduction_percent'];changes.append(dict(label=case['label'],before=before,after=after,reduction_percent=reduction))
 files=[Path(__file__),d/'build/report.json',check/'report.json',d/'build/diagnostic.wasm']+[ROOT/p for p in identities]+replies
 result=dict(complete=True,all21_native_and_current_control_bits_equal=True,all59_saved_replies_redecoded=True,all34_current_control_WAT_equal=True,module=r['wasm_sha256'],largest_preparation_instructions=max(p['instructions']for p in r['preparations']),changes=changes,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},scope='Local projection component. Quantization/input preparation/projection measured; immutable preparation reported separately. No full paid inference or capacity adoption claim.')
 (d/'post-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[d/'post-audit.json']):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps({k:v for k,v in result.items()if k not in ['source_hashes','changes']}))
if __name__=='__main__':main()
