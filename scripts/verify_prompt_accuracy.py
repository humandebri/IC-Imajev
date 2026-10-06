#!/usr/bin/env python3
"""Audit completed paired accuracy reports against their frozen input and source."""
import json,pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"client"))
from decision_validation import validate_decision
import evaluate_prompt_accuracy as e
D=e.ROOT/'artifacts/prompt-accuracy-v2'
def main():
 r=json.loads((D/'report.json').read_text());f=json.loads((D/'inputs.json').read_text())
 assert r['complete'] and r['completed_inferences']==64 and len(r['cases'])==64
 assert all(e.sha(e.ROOT/k)==v for k,v in r['source_hashes'].items()),'source changed'
 rows=[];counts={'original':0,'short':0};fallbacks=[]
 for pair in f['pairs']:
  for variant in ['original','short']:
   d=D/'runs'/f'{pair["id"]}-o{pair["offset"]}-{variant}';report=json.loads((d/'report.json').read_text());record=f['records'][pair[variant+'_record']]
   assert report['deployed_wasm_sha256']==e.MODULE
   assert report['model']==json.loads((e.ROOT/'checkpoints/full-int8.manifest.json').read_text())['model']
   validate_decision(report['decision_query']['ok']['decision'],record['options'])
   assert report['replayed_queries']==0 and report['executed_query_count']==report['query_count'],(pair['id'],variant)
   if report.get('fallback'):fallbacks.append((pair['id'],variant))
   assert not (d/'queries/failures.jsonl').exists() or report.get('fallback'),'unaccounted inference failure'
   row=e.extract(report,record,pair,variant,d);rows.append(row);counts[variant]+=1
 assert counts=={'original':32,'short':32}
 assert rows==r['cases']
 summary=e.summarize(f['pairs'],rows)
 for k,v in summary.items():assert r[k]==v,k
 assert len(summary['paired_comparisons'])==32 and len(summary['order_checks'])==16
 assert all(v['total']==24 for v in summary['primary_distinct_cases'].values())
 audit=dict(verified_inferences=64,paired_comparisons=32,primary_distinct_cases=24,order_checks=16,all_report_hashes_match=True,all_input_and_module_identities_match=True,all_typed_decisions_valid=True,no_replayed_inferences=True,source_hashes_match=True,fallbacks=fallbacks,report_sha256=e.sha(D/'report.json'))
 e.atomic_json(D/'verification.json',audit)
 print(json.dumps(dict(audit=audit,accuracy=summary['primary_distinct_cases'],regressions=summary['regressions'],improvements=summary['improvements'],answer_changes=summary['answer_changes'],order_changes=summary['order_changes']),ensure_ascii=False))
if __name__=='__main__':main()
