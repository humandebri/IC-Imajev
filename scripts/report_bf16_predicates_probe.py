#!/usr/bin/env python3
"""Audit shifted-bit max predicates, raw replies and IC body counters."""
from pathlib import Path
import hashlib,json,struct,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/bf16-predicates-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'check/report.json').read_text());b=json.loads((D/'build/report.json').read_text())
 control=json.loads((D/'control-source-audit.json').read_text());assert control['complete'] and control['control_bodies_identical_to_core'];assert all(sha(ROOT/p)==h for p,h in control['source_hashes'].items())
 assert r['complete'] and r['queries']==212 and len(r['cases'])==53 and r['module']==b['module']==sha(D/'build/diagnostic.wasm')
 for manifest in [r['source_hashes'],b['source_hashes'],b['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in manifest.items())
 upstream=ROOT/'scripts/check_finite_simd_probe.py';assert sha(upstream)==(D/'check-upstream.sha256').read_text().strip()
 helper=ROOT/'artifacts/f32-block-native/release/f32_args';improvements=[]
 for case in r['cases']:
  p=D/'check'/f"{case['label']}.input.bin";assert sha(p)==case['input_sha256'];raw=p.read_bytes();stride=struct.unpack_from('<I',raw)[0];bits=np.frombuffer(raw,dtype='<u4',offset=4)
  assert stride==case['stride'] and len(bits)==case['values']
  chunks=[bits[i:i+stride]for i in range(0,len(bits),stride)];all_bits=[int(np.all((x&65535)==0))for x in chunks]or[1];classified=[2 if not np.all(np.isfinite(x.view('<f4')))else int(np.all((x&65535)==0))for x in chunks]or[1]
  assert [v['method']for v in case['measurements']]==list(range(4))
  for row in case['measurements']:
   reply=D/'check'/f"{case['label']}-{row['method']}.hex";assert sha(reply)==row['reply_sha256']
   m=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));assert m==row['measurement'] and m['digest']==(all_bits if row['method']<2 else classified) and m['output_values']==len(bits)
   assert m['total_instructions']==m['input_prepare_instructions']+m['project_instructions']
  if case['label'].startswith('finite-')or case['label']=='real-617':
   counts=[v['measurement']['project_instructions']for v in case['measurements']]
   improvements.append(dict(label=case['label'],counts=counts,bf16_all_gain_percent=100*(1-counts[1]/counts[0]),bf16_classify_gain_percent=100*(1-counts[3]/counts[2])))
 files=[D/'control-source-audit.json',Path(__file__),ROOT/'scripts/build_bf16_predicates_probe.py',ROOT/'scripts/check_bf16_predicates_probe.py',upstream,D/'frozen-builder.py',D/'frozen-check.py',D/'check-upstream.sha256',D/'build/report.json',D/'check/report.json']+[ROOT/p for p in r['source_hashes']]+list((D/'check').glob('*.hex'));files=list(dict.fromkeys(files))
 summary=dict(complete=True,module=r['module'],cases=53,queries=212,all_predicates_equal=True,saved_candid_redecoded=True,control_bodies_identical_to_core=True,body_improvements=improvements,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='BF16 pure predicates only: original4 and unrolled64 summaries agree. Low-bit OR and absolute unsigned max unchanged, including invalid activation error precedence. All65536 BF16 patterns, arbitrary F32 bits and all nonfinite positions. Full inference unmeasured.')
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,body_improvements=improvements),indent=2))
if __name__=='__main__':main()
