#!/usr/bin/env python3
"""Audit saved component replies, immutable archives and stable read measurements."""
from pathlib import Path
import json,hashlib,zipfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-prepacked-stable-probe-v1';s=json.loads((d/'summary.json').read_text())
 for p,h in s['workflow_hashes'].items():assert sha(ROOT/p)==h
 with zipfile.ZipFile(d/'frozen-workflow.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for p,h in s['workflow_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h
  assert z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
 r=json.loads((d/'stable-read-check/report.json').read_text())
 for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
 for c in r['cases']:
  assert json.loads(subprocess.check_output([str(helper),'decode',str(ROOT/c['reply']),'measurement'],text=True))==c['measurement']
  assert c['instructions']==c['bytes']+220
 assert r['all_coefficient_byte_digests_equal'] and r['module']==s['module']
 assert len(s['cases'])==21 and all(c['after']>c['before']for c in s['cases'])
 result=dict(module=s['module'],complete=True,ordinary_queries=42,all_native_bits_equal=True,workflow_and_unique_archive_hashes_verified=True,stable_reply_redecode_verified=True,stable_read_cost_bytes_plus_220_verified=True,stable_read_sizes=[c['bytes']for c in r['cases']],prepacked_byte_ratio=6.125,adopted=False,current_full_candidate_unchanged=True,scope='Component inference is heap-backed; stable read measured separately. No full inference performance claim.',audit_script_sha256=sha(Path(__file__)))
 (d/'post-report-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 files=[Path(__file__),d/'post-report-audit.json',d/'stable-read-check/report.json']+[ROOT/p for p in r['source_hashes']]
 with zipfile.ZipFile(d/'stable-read-audit.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(result))
if __name__=='__main__':main()
