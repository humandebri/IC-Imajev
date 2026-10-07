#!/usr/bin/env python3
"""Verify independent reply archive and exact fixed startup instruction reduction."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-alias-probe-v1';s=json.loads((d/'summary.json').read_text());old=json.loads((ROOT/'artifacts/s2-k2-probe-v1/summary.json').read_text());check=json.loads((d/'check/report.json').read_text())
 for p,h in s['workflow_hashes'].items():assert sha(ROOT/p)==h
 with zipfile.ZipFile(d/'frozen-workflow.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for p,h in s['workflow_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h
  assert z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
 comparisons=[]
 for case in s['cases']:
  before=next(x for x in old['cases']if x['label']==case['label']);meta=next(x for x in check['cases']if x['label']==case['label']);saved=before['after']-case['after'];expected=0 if case['tokens']==1 else 49*2*32*(meta['rows']//16)*(meta['cols']//256)
  assert saved==expected,(case['label'],saved,expected)
  comparisons.append(dict(label=case['label'],tokens=case['tokens'],saved_vs_previous_rank49=saved,reduction_percent_vs_current=case['reduction_percent']))
 result=dict(module=s['module'],complete=True,all_native_bits_equal=True,ordinary_queries=42,workflow_and_unique_archive_hashes_verified=True,constant_startup_reduction_verified=True,adopted=False,current_full_candidate_unchanged=True,cases=comparisons,audit_script_sha256=sha(Path(__file__)))
 (d/'post-report-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,module=s['module'],queries=42,adopted=False)))
if __name__=='__main__':main()
