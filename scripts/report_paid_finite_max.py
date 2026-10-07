#!/usr/bin/env python3
"""Audit full paid finite-max proof, including independently reconstructed hidden30."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/paid-finite-max-v1';p=ROOT/'scripts/report_paid_quantize_cached_v2.py'
 s=p.read_text().replace('artifacts/paid-quantize-cached-v2','artifacts/paid-finite-max-v1')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 host=ROOT/'artifacts/terminal-mlp-reference-v1';reference=json.loads((host/'summary.json').read_text())
 assert reference['complete'] and reference['all_32_hidden_verified']
 assert all(sha(ROOT/p)==h for p,h in reference['workflow_hashes'].items())
 proof=json.loads((d/'proof/report.json').read_text());summary=json.loads((d/'summary.json').read_text());extras=[Path(__file__),p,host/'summary.json',host/'reconstruction/report.json',ROOT/'artifacts/update-finite-max-v1/runtime-comparison.json']
 hidden30=[]
 for item in proof['results']:
  row=next(v for v in json.loads((host/'reconstruction/report.json').read_text())['results']if v['case']==item['case'] and v['layer']==30)
  path=ROOT/row['output'];assert sha(path)==row['output_sha256'];a=np.load(path,allow_pickle=False);offset=item['quote']['prefix_tokens']-27
  assert offset>=0 and hashlib.sha256(a[offset:].astype('<f4').tobytes()).hexdigest()==item['debug']['hidden_hashes'][30]
  hidden30.append(dict(case=item['case'],layer=30,bit_equal=True,reference_kind='independent continuation of saved exact carry',suffix_offset=offset));extras.append(path)
 comparison=json.loads((ROOT/'artifacts/update-finite-max-v1/runtime-comparison.json').read_text());assert comparison==dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)
 summary.update(all_32_hidden_verified=True,hidden30=hidden30,scope='Actual paid API, three inputs, 31 historical exported hidden and independently reconstructed saved-carry hidden30. 32 conv/KV hashes, final norm and decision/probability F32 bits. Full worker totals include scheduler. Dense Delta arithmetic kernels unchanged from previous independently verified capture module.')
 summary['workflow_hashes'].update({str(p.relative_to(ROOT)):sha(p)for p in extras});(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Base reporters append summary entries. Freeze only the latest entry per name.
 archive=d/'frozen-paid-proof.zip';temporary=d/'frozen-paid-proof-deduplicated.zip'
 with zipfile.ZipFile(archive)as old,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED)as new:
  latest={v.filename:v for v in old.infolist()}
  for name,info in latest.items():
   if name!=str((d/'summary.json').relative_to(ROOT)):new.writestr(name,old.read(info))
  present=set(latest)
  for p in extras:
   name=str(p.relative_to(ROOT))
   if name not in present:new.write(p,name);present.add(name)
  new.write(d/'summary.json',str((d/'summary.json').relative_to(ROOT)))
 temporary.replace(archive)
 with zipfile.ZipFile(archive)as z:assert len(z.namelist())==len(set(z.namelist())) and z.read(str((d/'summary.json').relative_to(ROOT)))==(d/'summary.json').read_bytes()
 print(json.dumps(dict(all_32_hidden_verified=True,cases=summary['cases'],all_targets_met=summary['all_targets_met']),indent=2))
if __name__=='__main__':main()
