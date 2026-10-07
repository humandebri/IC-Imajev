#!/usr/bin/env python3
"""Verify full candidate differs only in eight predicate-dominated WAT bodies."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/update-k2-pair-guard-fold-v1';old=ROOT/'artifacts/update-k2-odd-roots-v1';k=ROOT/'artifacts/k2-pair-guard-fold-kernels-v1';probe=ROOT/'artifacts/k2-pair-guard-fold-probe-v1';r=json.loads((d/'build/report.json').read_text());o=json.loads((old/'build/report.json').read_text());kr=json.loads((k/'report.json').read_text());pr=json.loads((probe/'summary.json').read_text())
 assert pr['complete']and pr['all_native_bits_equal']and pr['ordinary_queries']==21
 assert len(r['patches'])==len(o['patches'])==30
 assert [(p['export'],p['source_sha256'])for p in r['patches'][:22]]==[(p['export'],p['source_sha256'])for p in o['patches'][:22]]
 for patch in r['patches'][22:]:assert any(patch['export']==x['symbol']and patch['source_sha256']==sha(ROOT/x['path'])for x in kr['kernels'])
 unchanged=[]
 for folder in ['build/runtime','build']:
  for p in (old/folder).glob('*.rs'):assert p.read_bytes()==(d/folder/p.name).read_bytes();unchanged.append(str(p.relative_to(old)))
 identities={}
 for report in [r,o,kr]:
  for key in ['source_hashes','dependency_hashes']:
   for path,h in report.get(key,{}).items():assert sha(ROOT/path)==h;identities[path]=h
 for path,h in pr['workflow_hashes'].items():assert sha(ROOT/path)==h;identities[path]=h
 assert sha(d/'build/full.wasm')==r['wasm_sha256']
 files=[Path(__file__),d/'build/report.json',d/'workflow-hashes.json',d/'entry-builder-hashes.json',old/'build/report.json',k/'report.json',probe/'summary.json']+[ROOT/p for p in identities]
 report=dict(module=r['wasm_sha256'],all_original_22_wat_sources_identical=True,all_runtime_and_wrapper_rs_byte_equal=unchanged,component_native_bits_equal=True,component_conditions=21,first_pair_byte_unchanged=kr['first_pair_byte_unchanged'],guard_conditions=kr['guard_conditions'],full_paid_fidelity_verified=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},scope='Source and component verification; full paid and Dense fidelity remain required.')
 (d/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'source-audit.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in dict.fromkeys(files+[d/'source-audit.json']):z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(d/'source-audit.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for path,h in report['source_hashes'].items():assert hashlib.sha256(z.read(path)).hexdigest()==h
 print(json.dumps(dict(module=report['module'],runtime_wrapper_equal=True,archive_unique=True)))
if __name__=='__main__':main()
