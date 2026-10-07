#!/usr/bin/env python3
"""Re-decode all finite predicate replies against the saved integer-bit oracle."""
from pathlib import Path
import hashlib,json,struct,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/finite-simd-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'check/report.json').read_text());assert r['complete'] and r['queries']==212 and len(r['cases'])==53
 assert all(sha(ROOT/p)==h for p,h in r['source_hashes'].items());helper=ROOT/'artifacts/f32-block-native/release/f32_args';improvements=[]
 for case in r['cases']:
  p=D/'check'/f"{case['label']}.input.bin";assert sha(p)==case['input_sha256'];raw=p.read_bytes();stride=struct.unpack_from('<I',raw)[0];bits=np.frombuffer(raw,dtype='<u4',offset=4);assert stride==case['stride'] and len(bits)==case['values']
  expected=[int(np.all(((bits[i:i+stride]>>23)&255)!=255))for i in range(0,len(bits),stride)]or[1]
  for row in case['measurements']:
   reply=D/'check'/f"{case['label']}-{row['method']}.hex";assert sha(reply)==row['reply_sha256']
   m=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));assert m==row['measurement'];assert m['digest']==expected;assert m['total_instructions']==m['input_prepare_instructions']+m['project_instructions']
  if case['label'].startswith('finite-')or case['label']=='real-617':
   counts=[x['measurement']['project_instructions']for x in case['measurements']];improvements.append(dict(label=case['label'],counts=counts,reductions=[100*(1/x[0])for x in []],simd64_reduction_percent=100*(1-counts[3]/counts[0])))
 files=[Path(__file__),ROOT/'scripts/build_finite_simd_probe.py',ROOT/'scripts/check_finite_simd_probe.py',D/'build/report.json',D/'check/report.json']+[ROOT/p for p in r['source_hashes']]+list((D/'check').glob('*.hex'));files=list(dict.fromkeys(files))
 summary=dict(complete=True,module=r['module'],cases=53,queries=212,all_predicates_equal=True,saved_candid_redecoded=True,selected_width=64,body_improvements=improvements,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Finite iff IEEE exponent differs from255. SIMD masks ignore sign/mantissa and preserve all scalar truth values. All65536 BF16 patterns, arbitrary F32 bits, signed zeros, subnormals, vector/tail NaNs/Inf. No full inference claim.')
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,cases=53,queries=212,selected_width=64)))
if __name__=='__main__':main()
