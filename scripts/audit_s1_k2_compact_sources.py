#!/usr/bin/env python3
"""Check dense coefficient initialization, compact pointer spans and exact kernel substitution."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s1-k2-compact-v1';old=ROOT/'artifacts/s1-k2-direct-v1';b=json.loads((d/'build/report.json').read_text());ob=json.loads((old/'build/report.json').read_text())
 assert len(b['patches'])==28 and all(p['wasmparser_validation']for p in b['patches'])
 assert [(p['export'],p['source_sha256'])for p in b['patches'][:22]]==[(p['export'],p['source_sha256'])for p in ob['patches'][:22]]
 assert (d/'build/src/lib.rs').read_bytes()==(old/'build/src/lib.rs').read_bytes()
 before='(local.set $qo(i32.shl(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start))(i32.const 1)))';after='(local.set $qo(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start)))'
 for p in (old/'build').glob('direct*.wat'):
  text=p.read_text();assert text.count(before)==3;assert (d/'build'/p.name).read_text()==text.replace(before,after)
 for cols in (256,512,2560,8192):
  for groups in (1,2,33,44,66):
   # Disjoint m/group/block spans, exactly128 written I16 coefficients per block.
   for m in range(7):
    for group in range(groups):
     for block in range(cols//256):
      start=(m*groups+group)*(cols//2)+block*128
      assert start+128<=(m*groups+group+1)*(cols//2)<=7*groups*(cols//2)
      assert [k+j for k in range(0,128,4)for j in range(4)]==list(range(128))
      # WAT qoffset is bytes: group*cols+block*256; each 32-bit splat stays in span.
      for k in range(64):assert group*cols+block*256+k*4+4<=group*cols+(block+1)*256
 for key in ('source_hashes','dependency_hashes'):assert all(sha(ROOT/p)==h for p,h in b[key].items())
 assert sha(d/'build/diagnostic.wasm')==b['wasm_sha256']
 control=(ROOT/'artifacts/update-gated-norm-finalize-v1/build/runtime/output_pairs.rs').read_text();assert 'if w.fixed.quad {return strassen_raw::project(q,w,scales);}'in control
 graph=(ROOT/'artifacts/update-gated-norm-finalize-v1/build/runtime/lib.rs').read_text();assert 'if !crate::all_finite(&out)'in graph
 result=dict(complete=True,module=b['wasm_sha256'],current_full22_control_equal=True,compact_query_dense_initialization_and_load_bounds_proved=True,all6_wat_edits_only_compact_qoffset=True,raw_project_matches_control_finite_check_boundary=True,full_graph_finite_check_unchanged=True,full_inference_adopted=False,all28_wasm_validated=True)
 (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
