#!/usr/bin/env python3
"""Independently re-decode every quantize reply and audit all complete scale/value bytes."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/quantize-cached-v1';HELPER=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'check/report.json').read_text());b=json.loads((D/'build/report.json').read_text());assert r['module']==b['module']==sha(D/'build/diagnostic.wasm')
 for manifest in [b['source_hashes'],b['dependency_hashes'],r['workflow_hashes']]:assert all(sha(ROOT/p)==h for p,h in manifest.items())
 assert (D/'build/baseline.rs').read_bytes()==(ROOT/'crates/imajev-runtime/src/quantize_simd.rs').read_bytes();assert len(r['cases'])==27 and r['queries']==81
 eps=Fraction(1,2**24);bound=127*(1+eps)/(1-eps);proof=json.loads((D/'normal-bound.json').read_text());assert proof['upper_quotient_fraction']==[bound.numerator,bound.denominator] and bound<Fraction(255,2)
 cases=[]
 for c in r['cases']:
  ip=D/'check'/f"{c['label']}.input.bin";ep=D/'check'/f"{c['label']}.oracle.bin";assert sha(ip)==c['input_hash'] and sha(ep)==c['oracle_hash'];values=np.frombuffer(ip.read_bytes(),dtype='<f4').reshape(-1,256);peak=(values.view('<u4')&0x7fffffff).max(axis=1);assert values.size//256==c['blocks']
  if np.any(peak>=0x7f800000):expected=b'ERR:integer projection input';assert c['error']
  else:
   with np.errstate(all='ignore'):
    maximum=peak.view('<f4');scale=np.divide(maximum,np.float32(127),dtype=np.float32);scale=np.maximum(scale,np.array([1],dtype='<u4').view('<f4')[0]);scale[peak==0]=np.float32(1);quant=np.minimum(np.maximum(np.rint(np.divide(values,scale[:,None],dtype=np.float32)),np.float32(-127)),np.float32(127)).astype('<i2');expected=scale.astype('<f4').tobytes()+quant.tobytes()
   assert not c['error']
  assert ep.read_bytes()==expected
  for m in c['measurements']:
   reply=ROOT/m['reply'];assert sha(reply)==m['reply_sha256'];raw=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True));assert bytes(raw.pop('digest'))==expected;assert all(m[k]==v for k,v in raw.items());assert raw['total_instructions']==raw['input_prepare_instructions']+raw['project_instructions']
  body=[m['project_instructions']for m in c['measurements']];cases.append(dict(label=c['label'],blocks=c['blocks'],error=c['error'],body_instructions=body,cached_gain_percent=100*(1-body[1]/body[0]),cached_normal_gain_percent=100*(1-body[2]/body[0])))
 summary=dict(module=r['module'],all_values_scales_and_errors_equal=True,normal_bound_verified=True,cases=cases,queries=81,whole_inference_verified=False,scope='Quantization component only; saved and generated F32 inputs, every complete I16 output and F32 scale byte compared. Input decode and digest serialization excluded from body interval.')
 paths=[Path(__file__),ROOT/'scripts/build_quantize_cached_probe.py',ROOT/'scripts/check_quantize_cached_probe.py',D/'normal-bound.json',D/'build/report.json',D/'check/report.json']+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'))+list((D/'check').glob('*.input.bin'))+list((D/'check').glob('*.oracle.bin'));paths=list(dict.fromkeys(paths));summary['workflow_hashes']={str(p.relative_to(ROOT)):sha(p)for p in paths};(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(queries=81,all_values_scales_and_errors_equal=True,shapes=[c for c in cases if c['label'].startswith('shape-')]),indent=2))
if __name__=='__main__':main()
