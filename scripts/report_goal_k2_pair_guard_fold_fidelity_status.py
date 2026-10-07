#!/usr/bin/env python3
"""Link completed latest paid and direct Dense fidelity evidence without claiming100B."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 paid=ROOT/'artifacts/paid-k2-pair-guard-fold-v1';d=ROOT/'artifacts/delta-capture-k2-pair-guard-fold-v1'
 paid_summary=json.loads((paid/'summary.json').read_text());c=json.loads((d/'summary.json').read_text());a=json.loads((d/'direct-reference-audit.json').read_text());s=json.loads((d/'source-audit.json').read_text());post=json.loads((paid/'post-report-audit.json').read_text())
 assert paid_summary['complete']and paid_summary['all_32_hidden_verified']and c['complete']and c['all32_hidden_verified']
 assert a['all72_complete_capture_bytes_equal_to_frozen_reference']and a['independent_ordered_recurrence_report_verified']and c['pre_bf_output_bits_equal']
 assert post['all_32_hidden_report_and_reference_hashes_verified']and post['raw_candid_calls_redecoded']>30
 assert s['paid_module']==paid_summary['module']and s['capture_module']==c['module']and s['all30_kernel_sources_equal']
 for summary in [paid_summary,c]:
  assert all(sha(ROOT/path)==h for path,h in summary['workflow_hashes'].items())
 assert c['baseline_restored']and post['baseline_restored']and a['temporary_snapshot_deleted']
 targets=all(x['total_handler_instructions']<=100_000_000_000 for x in paid_summary['cases']);assert paid_summary['all_targets_met']==targets
 files=[Path(__file__),paid/'summary.json',paid/'source-audit.json',paid/'post-report-audit.json',d/'summary.json',d/'source-audit.json',d/'direct-reference-audit.json']
 r=dict(complete_fidelity_evidence=True,all_targets_met=targets,goal_complete=False,performance_cases=paid_summary['cases'],paid_module=paid_summary['module'],dense_correctness_counterpart_module=c['module'],dense_state_values_checked=c['dense_state_values_checked'],full_capture_bytes_equal_to_frozen_reference=True,all32_hidden_verified=True,billing_and_performance_proof_separate=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 output=d/'goal-fidelity-status.json';output.write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'goal-fidelity-status.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[output]:z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(d/'goal-fidelity-status.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for path,h in r['source_hashes'].items():assert hashlib.sha256(z.read(path)).hexdigest()==h
 print(json.dumps(dict(complete_fidelity_evidence=True,goal_complete=False,all_targets_met=targets,max_handler_instructions=max(c['total_handler_instructions']for c in paid_summary['cases']))))
if __name__=='__main__':main()
