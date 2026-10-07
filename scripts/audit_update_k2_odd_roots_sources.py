#!/usr/bin/env python3
"""Verify WAT-only tail root specialization and freeze unique full build sources."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/update-k2-odd-roots-v1';old=ROOT/'artifacts/update-k2-compact-v1';r=json.loads((d/'build/report.json').read_text());o=json.loads((old/'build/report.json').read_text());k=ROOT/'artifacts/k2-odd-roots-kernels-v1';kr=json.loads((k/'report.json').read_text());integer=json.loads((k/'integer-audit.json').read_text());probe=ROOT/'artifacts/k2-odd-roots-probe-v1/summary.json';s=json.loads(probe.read_text());assert s['all_native_bits_equal']and s['ordinary_queries']==21
 assert len(r['patches'])==30 and all(p['wasmparser_validation']for p in r['patches']);assert [(p['export'],p['source_sha256'])for p in r['patches'][:22]]==[(p['export'],p['source_sha256'])for p in o['patches'][:22]]
 for p in r['patches'][22:]:assert any(p['export']==x['symbol']and p['source_sha256']==sha(ROOT/x['path'])for x in kr['kernels'])
 assert sha(d/'build/full.wasm')==r['wasm_sha256']
 for field in ['source_hashes','dependency_hashes']:
  for p,h in r[field].items():assert sha(ROOT/p)==h
 unchanged=[]
 for p in (old/'build/runtime').glob('*.rs'):assert p.read_bytes()==(d/'build/runtime'/p.name).read_bytes();unchanged.append(p.name)
 for p in (old/'build').glob('*.rs'):assert p.read_bytes()==(d/'build'/p.name).read_bytes()
 assert integer['all_integer_dots_equal'];native=old/'native-check/report.json';n=json.loads(native.read_text());assert n['native_bits_equal']and n['conditions']==300
 for report in [integer,n]:
  for p,h in report['source_hashes'].items():assert sha(ROOT/p)==h
 files=[Path(__file__),d/'build/report.json',d/'workflow-hashes.json',d/'frozen-builder.py',k/'integer-audit.json',native,probe]+[ROOT/p for p in r['source_hashes']]+[ROOT/p for p in integer['source_hashes']]+[ROOT/p for p in n['source_hashes']];files=list(dict.fromkeys(files));result=dict(module=r['wasm_sha256'],all_original_22_wat_sources_identical=True,all_runtime_rs_identical=unchanged,wrapper_rs_byte_exact=True,native_bits_equal=True,native_layout_conditions=300,independent_tail_integer_conditions=192,wasm_component_conditions=21,wasm_full_inference_fidelity_verified=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files});(d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(d/'source-audit.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'source-audit.json']:z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(d/'source-audit.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for p,h in result['source_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h
 print(json.dumps(dict(module=result['module'],unique_archive_verified=True,all_runtime_rs_identical=True)))
if __name__=='__main__':main()
