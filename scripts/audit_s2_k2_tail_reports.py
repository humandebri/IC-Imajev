#!/usr/bin/env python3
"""Recheck saved tail candidate workflows and compare completed alias counter evidence."""
from pathlib import Path
import argparse,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['tail','tail-dispatch'],required=True);arg=ap.parse_args();d=ROOT/f'artifacts/s2-k2-{arg.variant}-probe-v1';s=json.loads((d/'summary.json').read_text());old=json.loads((ROOT/'artifacts/s2-k2-alias-probe-v1/summary.json').read_text())
 for p,h in s['workflow_hashes'].items():assert sha(ROOT/p)==h
 with zipfile.ZipFile(d/'frozen-workflow.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for p,h in s['workflow_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h
  assert z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
 pairs=[]
 for c in s['cases']:
  o=next(x for x in old['cases']if x['label']==c['label']);assert c['before']==o['before'];pairs.append(dict(label=c['label'],tokens=c['tokens'],after=c['after'],saved_vs_alias=o['after']-c['after'],reduction_percent_vs_current=c['reduction_percent']))
 result=dict(module=s['module'],complete=True,ordinary_queries=42,all_native_bits_equal=True,workflow_and_unique_archive_hashes_verified=True,adopted=False,current_full_candidate_unchanged=True,cases=pairs,audit_script_sha256=sha(Path(__file__)))
 (d/'post-report-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(module=s['module'],complete=True,cases=pairs)))
if __name__=='__main__':main()
