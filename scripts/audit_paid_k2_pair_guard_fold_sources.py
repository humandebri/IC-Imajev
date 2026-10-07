#!/usr/bin/env python3
"""Audit K2 paid wrapper against proven canonical billing and owned scheduler."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/paid-k2-pair-guard-fold-v1';N=ROOT/'artifacts/update-k2-pair-guard-fold-v1';OLD=ROOT/'artifacts/paid-gated-norm-finalize-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=json.loads((D/'build/report.json').read_text());n=json.loads((N/'build/report.json').read_text());a=json.loads((N/'source-audit.json').read_text())
 assert a['all_original_22_wat_sources_identical']and a['component_native_bits_equal']
 assert b['wasm_sha256']==sha(D/'build/full.wasm') and b['update_candidate']==n['wasm_sha256']
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in b[key].items())
 assert len(b['patches'])==30 and all(p['wasmparser_validation']for p in b['patches'])
 assert [(x['export'],x['source_sha256'])for x in b['patches']]==[(x['export'],x['source_sha256'])for x in n['patches']]
 for name in ['lib.rs','update_inference.rs','paid_types.rs','paid_inference.rs']:
  assert (D/'build'/name).read_bytes()==(OLD/'build'/name).read_bytes(),name
 for name in ['paid_types.rs','paid_inference.rs']:assert (D/'build'/name).read_bytes()==(ROOT/'canisters/inference/src'/name).read_bytes()
 assert sha(N/'build/libimajev_runtime.rlib')==b['runtime_rlib_sha256']
 b.update(canonical_billing_sources_byte_equal=True,bound_prefix_and_bulk_digest=True,raw_packets_discarded_after_validation=True,paid_execution_verified=False)
 (D/'build/report.json').write_text(json.dumps(b,indent=2)+'\n')
 files=list(dict.fromkeys([Path(__file__),D/'build/report.json',N/'source-audit.json',D/'workflow-hashes.json']+[ROOT/p for p in b['source_hashes']]+[ROOT/p for p in a['source_hashes']]+[ROOT/p for p in json.loads((D/'workflow-hashes.json').read_text())]))
 result=dict(complete=True,module=b['wasm_sha256'],normal_module=n['wasm_sha256'],scheduler_byte_equal=True,canonical_billing_byte_equal=True,wrapper_byte_equal=True,original22_wat_equal=True,all30_patch_sources_equal_normal=True,source_and_dependency_hashes_verified=True,full_paid_performance_verified=False,goal_complete=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (D/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(D/'source-audit.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'source-audit.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,module=b['wasm_sha256'],scheduler_byte_equal=True,canonical_billing_byte_equal=True)))
if __name__=='__main__':main()
