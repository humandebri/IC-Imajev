#!/usr/bin/env python3
"""Independently audit candidate and completed identical-module control replies."""
from pathlib import Path
import json,hashlib,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/k2-odd-four-probe-v1';r=json.loads((d/'check/report.json').read_text());b=json.loads((d/'build/report.json').read_text());baseline=ROOT/'artifacts/s2-k2-probe-v1/check/report.json';old=json.loads(baseline.read_text());assert sha(baseline)==r['baseline_report_sha256'];assert b['base_module']==old['wasm_sha256'];assert sha(d/'build/diagnostic.wasm')==r['wasm_sha256']==b['wasm_sha256'];assert r['ordinary_queries']==r['saved_baseline_queries']==len(r['cases'])==21
 for report in [r,b]:
  for p,h in report['source_hashes'].items():assert sha(ROOT/p)==h
 for p,h in b['dependency_hashes'].items():assert sha(ROOT/p)==h
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';cases=[];files=[Path(__file__),d/'check/report.json',d/'build/report.json',d/'source-audit.json',baseline]
 for c in r['cases']:
  o=next(x for x in old['cases']if x['label']==c['label']);assert c['input_sha256']==o['input_sha256']and c['native']==o['native'];assert sha(d/'check'/f"{c['label']}.input.bin")==c['input_sha256']
  for key in ['s1_pair_bounds','rank49_k2']:
   m=c['measurements'][key];p=ROOT/m['reply'];assert sha(p)==m['reply_sha256'];raw=json.loads(subprocess.check_output([str(helper),'decode',str(p),'measurement'],text=True));assert all(m[k]==v for k,v in raw.items());assert raw['digest']==c['native']['digest'];assert raw['total_instructions']==sum(raw[k]for k in ['quantize_instructions','input_prepare_instructions','project_instructions']);files.append(p)
  before=c['measurements']['s1_pair_bounds']['total_instructions'];after=c['measurements']['rank49_k2']['total_instructions'];assert abs(c['reduction_percent']-100*(1-after/before))<1e-10
  cases.append(dict(label=c['label'],tokens=c['tokens'],before=before,after=after,saved=before-after,reduction_percent=c['reduction_percent']))
 files+= [ROOT/p for p in b['source_hashes']]+[ROOT/p for p in r['source_hashes']];files=list(dict.fromkeys(files));s=dict(complete=True,module=b['wasm_sha256'],base_module=b['base_module'],ordinary_queries=21,independently_redecoded_saved_baseline_queries=21,all_native_bits_equal=True,cases=cases,adopted=False,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Same original method3 diagnostic/runtime code. Only eight rank7 odd-tail WAT bodies differ. Q component only; no full paid inference claim.')
 (d/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(d/'frozen-workflow.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for p,h in s['workflow_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h
  assert z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
 print(json.dumps(dict(module=s['module'],native_bits_equal=True,cases=cases)))
if __name__=='__main__':main()
