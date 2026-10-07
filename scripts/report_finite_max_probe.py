#!/usr/bin/env python3
"""Audit shifted-bit max predicates, raw replies and IC body counters."""
from pathlib import Path
import hashlib,json,struct,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/finite-max-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'check/report.json').read_text());b=json.loads((D/'build/report.json').read_text())
 assert r['complete'] and r['queries']==318 and len(r['cases'])==53 and r['module']==b['module']==sha(D/'build/diagnostic.wasm')
 for manifest in [r['source_hashes'],b['source_hashes'],b['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in manifest.items())
 upstream=ROOT/'scripts/check_finite_simd_probe.py';assert sha(upstream)==(D/'check-upstream.sha256').read_text().strip()
 helper=ROOT/'artifacts/f32-block-native/release/f32_args';improvements=[]
 for case in r['cases']:
  p=D/'check'/f"{case['label']}.input.bin";assert sha(p)==case['input_sha256'];raw=p.read_bytes();stride=struct.unpack_from('<I',raw)[0];bits=np.frombuffer(raw,dtype='<u4',offset=4)
  assert stride==case['stride'] and len(bits)==case['values']
  expected=[int(np.all(((bits[i:i+stride]>>23)&255)!=255))for i in range(0,len(bits),stride)]or[1]
  assert [v['method']for v in case['measurements']]==list(range(6))
  for row in case['measurements']:
   reply=D/'check'/f"{case['label']}-{row['method']}.hex";assert sha(reply)==row['reply_sha256']
   m=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));assert m==row['measurement'] and m['digest']==expected and m['output_values']==len(bits)
   assert m['total_instructions']==m['input_prepare_instructions']+m['project_instructions']
  if case['label'].startswith('finite-')or case['label']=='real-617':
   counts=[v['measurement']['project_instructions']for v in case['measurements']]
   improvements.append(dict(label=case['label'],counts=counts,max64_gain_percent=100*(1-counts[4]/counts[3]),maxfull_gain_percent=100*(1-counts[5]/counts[3])))
 files=[Path(__file__),ROOT/'scripts/build_finite_max_probe.py',ROOT/'scripts/check_finite_max_probe.py',upstream,D/'frozen-builder.py',D/'frozen-check.py',D/'check-upstream.sha256',D/'build/report.json',D/'check/report.json']+[ROOT/p for p in r['source_hashes']]+list((D/'check').glob('*.hex'));files=list(dict.fromkeys(files))
 summary=dict(complete=True,module=r['module'],cases=53,queries=318,all_predicates_equal=True,saved_candid_redecoded=True,body_improvements=improvements,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Finite predicate component: unsigned (bits<<1) maximum >=0xff000000 iff any IEEE exponent255. Early64 and full scan tested. All65536 BF16 patterns, arbitrary F32 bits, every64/tail nonfinite position. No full inference measurement.')
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,body_improvements=improvements),indent=2))
if __name__=='__main__':main()
