#!/usr/bin/env python3
"""Verify inlined integer reconstruction against native and the previous exact kernel."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/report_s3_prepared_probe.py';s=p.read_text().replace('artifacts/s3-prepared-v1','artifacts/s3-winograd-v1').replace("'s3_prepared'","'s3_winograd'").replace("len(r['cases']) == 19 and r['ordinary_queries'] == 38","len(r['cases']) == 21 and r['ordinary_queries'] == 42").replace('ordinary_queries=38','ordinary_queries=42')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 d=ROOT/'artifacts/s3-winograd-v1';r=json.loads((d/'summary.json').read_text());old=ROOT/'artifacts/s3-inline-v1/summary.json';by={c['label']:c for c in json.loads(old.read_text())['cases']}
 for c in r['cases']:
  if c['label'] in by:
   a=by[c['label']];assert c['tokens']==a['tokens'];c.update(previous_prepared_instructions=a['after'],versus_previous_percent=100*(1-c['after']/a['after']))
 assert all(sha(ROOT/p)==h for p,h in json.loads((d/'entry-hashes.json').read_text()).items())
 bounds=json.loads((d/'integer-bounds.json').read_text())
 assert bounds['all_651_reconstruction_nodes_checked'] and bounds['all_i32_intermediates_fit'] and bounds['all_final_roots_are_original_dots']
 assert len(bounds['bounds'])==651 and max(v['bound'] for v in bounds['bounds'])==bounds['max_canonical_reconstruction_bound']<2**31
 assert bounds['max_leaf_bound']<2**31
 assert all(sha(ROOT/p)==h for p,h in bounds['source_hashes'].items())
 r['integer_bounds_verified']=True
 r['max_leaf_bound']=bounds['max_leaf_bound']
 r['max_reconstruction_bound']=bounds['max_canonical_reconstruction_bound']
 r['previous_summary_sha256']=sha(old)
 files=[Path(__file__),ROOT/'scripts/build_s3_winograd_probe.py',ROOT/'scripts/check_s3_winograd_probe.py',ROOT/'scripts/report_s3_prepared_probe.py',ROOT/'scripts/check_winograd_integer_bounds.py',d/'ring-proof.json',d/'integer-bounds.json',d/'check-upstream.sha256',d/'entry-hashes.json',d/'frozen-builder.py',d/'frozen-generator.py',d/'frozen-check.py',d/'generated/generator.json',d/'generated/kernel.wat',d/'build/report.json',d/'check/report.json']
 files += [ROOT/p for p in json.loads((d/'build/report.json').read_text())['source_hashes']]
 files=list(dict.fromkeys(files))
 r['workflow_hashes']={str(p.relative_to(ROOT)):sha(p) for p in files};r['scope']='Q projection only, exact Winograd rank343 with inline reconstruction and unchanged FP block order. Canonical bilinear coefficient bounds cover every I32 leaf and reconstruction intermediate; verify against same-module S1 and independent native integers.'
 (d/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[d/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print('21 cases /42 replies verified against independent native integers')
if __name__=='__main__':main()
