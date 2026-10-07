#!/usr/bin/env python3
"""Audit shifted-bit max predicates, raw replies and IC body counters."""
from pathlib import Path
import hashlib,json,struct,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/positive-finite-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'check/report.json').read_text());b=json.loads((D/'build/report.json').read_text())
 assert r['complete'] and r['queries']==159 and len(r['cases'])==53 and r['module']==b['module']==sha(D/'build/diagnostic.wasm')
 for manifest in [r['source_hashes'],b['source_hashes'],b['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in manifest.items())
 upstream=ROOT/'scripts/check_finite_simd_probe.py';assert sha(upstream)==(D/'check-upstream.sha256').read_text().strip()
 helper=ROOT/'artifacts/f32-block-native/release/f32_args';improvements=[]
 for case in r['cases']:
  p=D/'check'/f"{case['label']}.input.bin";assert sha(p)==case['input_sha256'];raw=p.read_bytes();stride=struct.unpack_from('<I',raw)[0];bits=np.frombuffer(raw,dtype='<u4',offset=4)
  assert stride==case['stride'] and len(bits)==case['values']
  expected=[int(np.all(np.isfinite(bits[i:i+stride].view('<f4')) & (bits[i:i+stride].view('<f4')>0)))for i in range(0,len(bits),stride)]or[1]
  assert [v['method']for v in case['measurements']]==list(range(3))
  for row in case['measurements']:
   reply=D/'check'/f"{case['label']}-{row['method']}.hex";assert sha(reply)==row['reply_sha256']
   m=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));assert m==row['measurement'] and m['digest']==expected and m['output_values']==len(bits)
   assert m['total_instructions']==m['input_prepare_instructions']+m['project_instructions']
  if case['label'].startswith('finite-')or case['label']=='real-617':
   counts=[v['measurement']['project_instructions']for v in case['measurements']]
   improvements.append(dict(label=case['label'],counts=counts,max64_gain_percent=100*(1-counts[1]/counts[0]),maxfull_gain_percent=100*(1-counts[2]/counts[0])))
 files=[Path(__file__),ROOT/'scripts/build_positive_finite_probe.py',ROOT/'scripts/check_positive_finite_probe.py',upstream,D/'frozen-builder.py',D/'frozen-check.py',D/'check-upstream.sha256',D/'build/report.json',D/'check/report.json']+[ROOT/p for p in r['source_hashes']]+list((D/'check').glob('*.hex'));files=list(dict.fromkeys(files))
 summary=dict(complete=True,module=r['module'],cases=53,queries=159,all_predicates_equal=True,saved_candid_redecoded=True,body_improvements=improvements,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Positive finite component: unsigned (bits-1) maximum below0x7f7fffff iff all values strictly positive and finite. Independent floating predicate oracle and scalar Wasm control. All65536 BF16 patterns, arbitrary F32 bits, zero/negative/nonfinite positions. No full inference measurement.')
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,body_improvements=improvements),indent=2))
if __name__=='__main__':main()
