#!/usr/bin/env python3
"""Re-decode saved arithmetic replies and pin the complete probe workflow."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/add-norm-simd-v1';b=json.loads((d/'build/report.json').read_text());r=json.loads((d/'check/report.json').read_text())
 assert sha(d/'build/diagnostic.wasm')==b['module']==r['module']
 for g in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in b[g].items())
 helper=ROOT/'artifacts/f32-block-native/release/f32_args';assert sha(helper)==r['helper_sha256']
 assert sha(ROOT/'scripts/check_add_norm_simd_probe.py')==r['source_sha256']
 assert len(r['cases'])==11 and r['all_bits_equal']
 for c in r['cases']:
  assert sha(d/'check'/(c['label']+'.input.bin'))==c['input_sha256']
  expected=d/'check'/(c['label']+'.expected.bin');assert sha(expected)==c['expected_sha256']
  for method,m in enumerate(c['measurements']):
   reply=d/'check'/f'{c["label"]}-{method}.hex';assert sha(reply)==m['reply_sha256']
   decoded=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));assert bytes(decoded.pop('digest'))==expected.read_bytes();assert decoded==m['measurement']
 paths=[Path(__file__),ROOT/'scripts/build_add_norm_simd_probe.py',ROOT/'scripts/check_add_norm_simd_probe.py',d/'build/lib.rs',d/'build/add_norm_simd.rs',d/'build/report.json',d/'check/report.json']
 result=dict(module=r['module'],raw_replies_verified=True,all_bits_equal=True,cases=r['cases'],workflow_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},scope=r['scope'])
 (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in paths+[d/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print('11 cases /22 queries; all output bits and saved replies verified')
if __name__=='__main__':main()
