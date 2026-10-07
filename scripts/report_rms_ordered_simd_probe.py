#!/usr/bin/env python3
"""Recheck saved RMS outputs, counters and the frozen generator/checker provenance."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/report_add_norm_simd_probe.py';s=p.read_text().replace('add-norm-simd-v1','rms-ordered-simd-v1').replace('add_norm_simd_probe','rms_ordered_simd_probe').replace('add_norm_simd.rs','rms_ordered_simd.rs').replace("len(r['cases'])==11","len(r['cases'])==14").replace('11 cases /22 queries','14 cases /28 queries')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 d=ROOT/'artifacts/rms-ordered-simd-v1';r=json.loads((d/'summary.json').read_text())
 for n in ['entry-hashes.json','check-entry-hashes.json']:assert all(sha(ROOT/v)==h for v,h in json.loads((d/n).read_text()).items())
 files=[ROOT/v for v in r['workflow_hashes']]+[d/n for n in ['entry-hashes.json','check-entry-hashes.json','frozen-builder.py','frozen-check.py']]
 r['workflow_hashes']={str(v.relative_to(ROOT)):sha(v) for v in files};(d/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for v in files+[d/'summary.json']:z.write(v,str(v.relative_to(ROOT)))
if __name__=='__main__':main()
