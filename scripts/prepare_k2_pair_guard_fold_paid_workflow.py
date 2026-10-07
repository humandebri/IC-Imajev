#!/usr/bin/env python3
"""Freeze the proven paid workflow for the source-audited guard-fold candidate."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 files=[]
 for prefix in ['build_paid','audit_paid','prove_paid','report_paid']:
  suffixes={'build_paid':[''],'audit_paid':['_sources','_report'],'prove_paid':['_upgrade_guards',''],'report_paid':['']}[prefix]
  for suffix in suffixes:
   old=ROOT/'scripts'/f'{prefix}_k2_odd_roots{suffix}.py';new=ROOT/'scripts'/f'{prefix}_k2_pair_guard_fold{suffix}.py';assert not new.exists()
   s=old.read_text().replace('k2-odd-roots-v1','k2-pair-guard-fold-v1')
   if suffix=='_sources' or prefix=='build_paid':s=s.replace("audit['native_bits_equal']","audit['component_native_bits_equal']").replace("a['native_bits_equal']","a['component_native_bits_equal']")
   new.write_text(s);files.extend([old,new])
 d=ROOT/'artifacts/k2-pair-guard-fold-paid-workflow-v1';d.mkdir(exist_ok=False);files.append(Path(__file__))
 (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 print(json.dumps(dict(generated_scripts=6,paid_and_dense_fidelity_pending=True)))
if __name__=='__main__':main()
