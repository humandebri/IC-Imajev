#!/usr/bin/env python3
"""Prepare the separate complete Dense correctness workflow after paid proof."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 names=['build_delta_capture_k2_odd_roots','audit_delta_capture_k2_odd_roots_sources','prove_delta_capture_k2_odd_roots','report_delta_capture_k2_odd_roots','audit_delta_capture_k2_odd_roots_reference','report_goal_k2_odd_roots_fidelity_status'];files=[Path(__file__)]
 for name in names:
  old=ROOT/'scripts'/(name+'.py');new=ROOT/'scripts'/(name.replace('k2_odd_roots','k2_pair_guard_fold')+'.py');assert not new.exists()
  text=old.read_text().replace('k2-odd-roots-v1','k2-pair-guard-fold-v1').replace('k2_odd_roots','k2_pair_guard_fold');new.write_text(text);files.extend([old,new])
 d=ROOT/'artifacts/k2-pair-guard-fold-dense-workflow-v1';d.mkdir(exist_ok=False);(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n');print(json.dumps(dict(generated_scripts=6,full_capture_execution_pending=True)))
if __name__=='__main__':main()
