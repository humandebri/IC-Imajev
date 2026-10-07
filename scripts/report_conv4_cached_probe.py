#!/usr/bin/env python3
"""Verify cached-table replies and compare the complete kernel with prior SIMD."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/report_conv4_probe.py';s=p.read_text().replace('artifacts/conv4-simd-v3','artifacts/conv4-cached-v1').replace("'scripts/check_conv4_probe.py'","'scripts/check_conv4_cached_probe.py'")
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 d=ROOT/'artifacts/conv4-cached-v1';r=json.loads((d/'summary.json').read_text());old=ROOT/'artifacts/conv4-simd-v3/summary.json';by_name={c['label']:c for c in json.loads(old.read_text())['cases']}
 for c in r['cases']:
  prev=by_name[c['label']];assert c['input_sha256']==prev['input_sha256'] and c['preactivation_sha256']==prev['preactivation_sha256']
  before=prev['measurements'][1]['measurement']['project_instructions'];after=c['measurements'][1]['measurement']['project_instructions']
  c.update(previous_simd_instructions=before,versus_previous_simd_percent=100*(1-after/before))
 r['previous_summary_sha256']=sha(old)
 for p in [ROOT/'scripts/build_conv4_cached_probe.py',ROOT/'scripts/check_conv4_cached_probe.py',ROOT/'scripts/check_conv4_probe.py',ROOT/'scripts/report_conv4_probe.py',d/'frozen-check.py',d/'build/runtime/prepared_activation.rs']:r['workflow_hashes'][str(p.relative_to(ROOT))]=sha(p)
 (d/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in list(r['workflow_hashes'])+[str((d/'summary.json').relative_to(ROOT))]:z.write(ROOT/p,p)
 print('Cached table: 10 cases /40 replies and independent preactivation verified')
if __name__=='__main__':main()
