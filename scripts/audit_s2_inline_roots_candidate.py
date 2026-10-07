#!/usr/bin/env python3
"""Prove root aliases remain live and screen copy-only improvement against prior IC."""
from pathlib import Path
import hashlib,json,re,runpy,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-inline-roots-kernels-v1';r=json.loads((d/'report.json').read_text())
 for p,h in r['source_hashes'].items():assert sha(ROOT/p)==h,p
 helper=runpy.run_path(str(ROOT/'scripts/generate_s2_inline_roots_kernels.py'))
 plan=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py';ns={'__name__':'plan','__file__':str(plan)};exec(compile(plan.read_text(),str(plan),'exec'),ns);_,_,c,_,roots,_=ns['plan']();assert ns['reconstruct'](c,roots)==helper['reconstruction'](c,roots)
 files=[Path(__file__),d/'report.json',plan]+[ROOT/p for p in r['source_hashes']];checked=0
 pattern=re.compile(r'(?:local.get \$\w+\nlocal.set \$c[0-3]\n){4}')
 for k in r['kernels']:
  candidate=ROOT/k['path'];original=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1'/candidate.name;old=original.read_text();new=candidate.read_text();assert helper['remove_copies'](old)[0]==new
  matches=list(pattern.finditer(old));assert len(matches)==k['root_copy_groups_removed']
  for i,match in enumerate(matches):
   end=matches[i+1].start()if i+1<len(matches)else len(old);segment=old[match.end():end];bindings=re.findall(r'local.get (\$\w+)\nlocal.set (\$c[0-3])',match[0])
   assert len(re.findall(r'local.get \$c[0-3]\b',segment))==8
   for source,alias in bindings:
    uses=list(re.finditer(r'local.get '+re.escape(alias)+r'\b',segment));assert uses
    before_last=segment[:uses[-1].end()];assert not re.search(r'local\.(set|tee) '+re.escape(source)+r'\b',before_last),source;checked+=1
  files.extend([original,candidate])
 measured=ROOT/'artifacts/s2-k2-leaf-outer-probe-v1/check/report.json';old=json.loads(measured.read_text());files.append(measured);predictions=[]
 for case in old['cases']:
  if case['label']not in ['617','size-48','size-56','size-57','normal']:continue
  m=case['measurements'];before=m['current_guard_fold_k2']['total_instructions'];candidate=m['rank49_leaf_outer']['total_instructions'];count=(case['rows']//16)*4*8*((case['tokens']+3)//4)*(case['cols']//256)
  predictions.append(dict(label=case['label'],old_current=before,old_rank49=candidate,removed_local_instructions=count,predicted_copy_only=candidate-count,predicted_still_above_current=candidate-count>before))
 assert all(p['predicted_still_above_current']for p in predictions if p['label']!='normal')
 report=dict(complete=True,root_alias_liveness_checks=checked,C_reconstruction_byte_equal=True,all_changed_operations_are_root_local_copies=True,predictions=predictions,wasm_execution_verified=False,performance_verified=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},scope='Alias proof plus estimate using previously measured unit local-op deltas. Predictions are not actual new IC counters; copy-only change cannot close observed goal-shape gap under that model. No adopted/full inference claim.')
 (d/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'audit-evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[d/'audit.json']):z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,root_alias_liveness_checks=checked,predictions=predictions)))
if __name__=='__main__':main()
