#!/usr/bin/env python3
"""Explain all observed same-module old/selective stream counter deltas."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s3-selective-streaming-paired-probe-v1';report=d/'check/report.json';r=json.loads(report.read_text());audit=json.loads((d/'post-audit.json').read_text());assert audit['complete']
 for p,h in audit['source_hashes'].items():assert sha(ROOT/p)==h,p
 old=json.loads((ROOT/'artifacts/rank343-streaming-reconstruction-v1/report.json').read_text());new=json.loads((ROOT/'artifacts/rank343-selective-streaming-inline-v1/gap-4/report.json').read_text())
 nodes_old=sum(e['kind']=='reconstruct'for e in old['events']);nodes_new=sum(e['kind']=='reconstruct'for e in new['events']);assert(nodes_old,nodes_new)==(700,328)
 lines=old['reconstruction_instruction_lines']-new['reconstruction_instruction_lines'];assert lines==744
 cases=[]
 for c in r['cases']:
  m=c['measurements'];actual=m['rank343_k2']['total_instructions']-m['selective_streaming_k2']['total_instructions'];tiles=c['rows']//256;blocks=c['cols']//256;loops=(c['tokens']+87)//88;octets=(c['tokens']+7)//8
  prediction=tiles*blocks*((nodes_old-nodes_new)*11*7*loops+lines*8*octets);assert actual==prediction,c['label']
  assert m['rank343_k2']['quantize_instructions']==m['selective_streaming_k2']['quantize_instructions'] and m['rank343_k2']['input_prepare_instructions']==m['selective_streaming_k2']['input_prepare_instructions']
  assert m['rank343_k2']['project_instructions']-m['selective_streaming_k2']['project_instructions']==actual
  assert m['selective_streaming_k2']['total_instructions']>m['current_guard_fold_k2']['total_instructions']
  cases.append(dict(label=c['label'],actual_delta=actual,predicted_delta=prediction,old_selective_projection_only=True))
 files=[Path(__file__),report,d/'post-audit.json',ROOT/'artifacts/rank343-streaming-reconstruction-v1/report.json',ROOT/'artifacts/rank343-selective-streaming-inline-v1/gap-4/report.json']
 result=dict(complete=True,conditions=len(cases),cases=cases,removed_C_nodes=372,removed_C_instruction_lines=744,formula='(rows/256)*(cols/256)*(28644*ceil(tokens/88)+5952*ceil(tokens/8))',all21_selective_still_worse_than_current=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Exact fit to all21 observed same-module counter differences, accounted by removed always-tested C guards and active integer expression lines. Not a universal opcode meter or full paid claim.')
 (d/'delta-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'delta-evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'delta-audit.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,conditions=len(cases),all21_deltas_exact=True,selective_adopted=False)))
if __name__=='__main__':main()
