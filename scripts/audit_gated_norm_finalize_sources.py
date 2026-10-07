#!/usr/bin/env python3
"""Reverse the only runtime edits and verify scheduler, billing, kernels and build hashes."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_build(d):
 b=json.loads((d/'build/report.json').read_text())
 for key in ('source_hashes','dependency_hashes'):assert all(sha(ROOT/p)==h for p,h in b[key].items())
 assert len(b['patches'])==22 and all(p['wasmparser_validation']for p in b['patches'])
 return b

def main():
 d=ROOT/'artifacts/update-gated-norm-finalize-v1';old=ROOT/'artifacts/update-attention-key-lanes-v1';b=verify_build(d)
 helper=(d/'gated_norm_finalize.rs').read_text();expected=(ROOT/'scripts/gated_norm_finalize.rs').read_text().replace('gate:Vec<f32>','gate:&[f32]').replace('.zip(gate)', '.zip(gate.iter().copied())');assert helper==expected
 edits=json.loads((d/'runtime-edits.json').read_text())
 for p in (old/'build/runtime').glob('*.rs'):
  text=(d/'build/runtime'/p.name).read_text()
  if p.name=='lib.rs':
   assert text.endswith(helper);text=text[:-len(helper)]
   for e in reversed(edits):assert text.count(e['after'])==1;text=text.replace(e['after'],e['before'])
  assert text==p.read_text(),p
 wat=list(old.glob('*.wat'))
 ob=json.loads((old/'build/report.json').read_text());assert [(p['export'],p['source_sha256'])for p in b['patches']]==[(p['export'],p['source_sha256'])for p in ob['patches']]
 for p in wat:assert p.read_bytes()==(d/p.name).read_bytes()
 result=dict(complete=True,normal_module=b['wasm_sha256'],runtime_reversal_equal=True,helper_matches_measured_component_except_borrowed_gate_signature=True,original22_wat_equal=True,source_and_dependency_hashes_verified=True,goal_complete=False)
 (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 paid=ROOT/'artifacts/paid-gated-norm-finalize-v1'
 if (paid/'build/report.json').exists():
  pb=verify_build(paid);po=ROOT/'artifacts/paid-attention-key-lanes-v1'
  for name in ('update_inference.rs','paid_types.rs','paid_inference.rs'):
   assert (paid/'build'/name).read_bytes()==(po/'build'/name).read_bytes(),name
  assert [(p['export'],p['source_sha256'])for p in pb['patches']]==[(p['export'],p['source_sha256'])for p in b['patches']]
  result=dict(complete=True,module=pb['wasm_sha256'],normal_module=b['wasm_sha256'],scheduler_byte_equal=True,canonical_billing_byte_equal=True,original22_wat_equal=True,source_and_dependency_hashes_verified=True,full_paid_performance_verified=False,goal_complete=False)
  (paid/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
