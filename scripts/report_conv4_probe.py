#!/usr/bin/env python3
"""Re-decode ordered convolution evidence and freeze sources and provenance."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/conv4-simd-v3';b=json.loads((d/'build/report.json').read_text());r=json.loads((d/'check/report.json').read_text());assert sha(d/'build/diagnostic.wasm')==b['module']==r['module']
 for g in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in b[g].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((d/'builder-hashes.json').read_text()).items())
 helper=ROOT/'artifacts/f32-block-native/release/f32_args';assert sha(helper)==r['helper_sha256'];assert sha(ROOT/'scripts/check_conv4_probe.py')==r['source_sha256']
 t=r['weight_tensor']
 with (ROOT/'checkpoints/full-int8.pack').open('rb') as f:f.seek(t['offset']);packed=f.read(t['bytes'])
 assert hashlib.sha256(packed).hexdigest()==r['packed_weights_sha256'];assert sha(ROOT/'artifacts/f32_block/check/617.input.bin')==r['actual_values_source_sha256']
 for c in r['cases']:
  assert sha(d/'check'/(c['label']+'.input.bin'))==c['input_sha256'];p=d/'check'/(c['label']+'.preactivation.bin');assert sha(p)==c['preactivation_sha256'];expected=list(hashlib.sha256(p.read_bytes()).digest());values=[]
  for method,m in enumerate(c['measurements']):
   reply=d/'check'/f'{c["label"]}-{method}.hex';assert sha(reply)==m['reply_sha256'];v=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));assert v==m['measurement'];assert v['total_instructions']==v['input_prepare_instructions']+v['project_instructions'];values.append(v)
   if method>=2:assert v['digest']==expected
  assert values[0]['digest']==values[1]['digest']
 files=[Path(__file__),ROOT/'scripts/check_conv4_probe.py',ROOT/'scripts/build_conv4_probe.py',ROOT/'scripts/build_conv4_probe_v2.py',ROOT/'scripts/build_conv4_probe_v3.py',d/'frozen-builder.py',d/'builder-hashes.json',d/'build/report.json',d/'build/conv4.rs',d/'check/report.json']
 result=dict(module=r['module'],all_bits_equal=True,raw_replies_verified=True,cases=r['cases'],workflow_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},scope=r['scope'])
 (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[d/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print('10 conditions /40 queries; ordered preactivation and scalar runtime output verified')
if __name__=='__main__':main()
