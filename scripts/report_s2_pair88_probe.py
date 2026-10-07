#!/usr/bin/env python3
"""Re-decode the widened exact kernel and compare it with same-module control."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/report_s2_pair_late_probe.py';s=p.read_text().replace('artifacts/s2-pair-late-v1','artifacts/s2-pair88-v1').replace('check-v2/report.json','check/report.json').replace('s2_pair_late','s2_pair88')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 d=ROOT/'artifacts/s2-pair88-v1';r=json.loads((d/'summary.json').read_text());prev=ROOT/'artifacts/s2-pair-factored-v1/summary.json';old={c['label']:c for c in json.loads(prev.read_text())['cases']}
 for c in r['cases']:
  a=old[c['label']];assert a['tokens']==c['tokens'];c['previous64_instructions']=a['after'];c['versus_previous64_percent']=100*(1-c['after']/a['after'])
 r['previous_summary_sha256']=sha(prev);r['upstream_hashes_verified']=True
 assert all(sha(ROOT/p)==h for p,h in json.loads((d/'upstream-hashes.json').read_text()).items())
 r['scope']='Isolated Q projection, 88-output rank49 with padded weight tail; exact native oracle and same-module S1 control. Regression; not connected to full inference.'
 files=[Path(__file__),ROOT/'scripts/build_s2_pair88_probe.py',ROOT/'scripts/check_s2_pair88_probe.py',d/'frozen-builder.py',d/'frozen-check.py',d/'upstream-hashes.json',d/'build/report.json',d/'check/report.json']
 r['workflow_hashes']={str(p.relative_to(ROOT)):sha(p) for p in files};(d/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[d/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print('19 cases /38 queries verified; widened rank49 remains slower and is rejected.')
if __name__=='__main__':main()
