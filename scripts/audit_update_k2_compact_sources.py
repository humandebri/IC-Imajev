#!/usr/bin/env python3
"""Verify isolated K2 integration and archive unique, current build evidence."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/update-k2-compact-v1';OLD=ROOT/'artifacts/update-gated-norm-finalize-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((D/'build/report.json').read_text());old=json.loads((OLD/'build/report.json').read_text());probe=ROOT/'artifacts/s1-k2-mlp160-v1';p=json.loads((probe/'build/report.json').read_text())
 assert len(r['patches'])==30 and all(x['wasmparser_validation']for x in r['patches'])
 assert [(x['export'],x['source_sha256'])for x in r['patches'][:22]]==[(x['export'],x['source_sha256'])for x in old['patches']]
 for patch in r['patches'][22:]:assert any((x['export'],x['source_sha256'])==(patch['export'],patch['source_sha256'])for x in p['patches'][22:])
 assert r['wasm_sha256']==sha(D/'build/full.wasm')
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in r[key].items())
 unchanged=[]
 for path in (OLD/'build/runtime').glob('*.rs'):
  if path.name=='strassen_raw.rs':continue
  assert path.read_bytes()==(D/'build/runtime'/path.name).read_bytes(),path.name;unchanged.append(path.name)
 # Runtime replacement is exactly the reviewed source plus original fail-closed exports.
 text=(OLD/'build/runtime/strassen_raw.rs').read_text();stubs=[]
 for line in text.splitlines():
  if line.startswith('unsafe extern "C" fn '):
   symbol=text[:text.index(line)].rsplit('#[export_name="',1)[1].split('"',1)[0]
   stubs.append('#[cfg(target_arch="wasm32")]\n#[export_name="'+symbol+'"]#[inline(never)]\n'+line)
 assert (D/'build/runtime/strassen_raw.rs').read_text()==(ROOT/'scripts/strassen_k2_runtime.rs').read_text()+'\n'+'\n'.join(stubs)+'\n'
 assert (D/'build/runtime/win_kernel.rs').read_bytes()==(D/'win_kernel.rs').read_bytes()
 for path in (OLD/'build').glob('*.rs'):assert path.read_bytes()==(D/'build'/path.name).read_bytes(),path.name
 n=json.loads((D/'native-check/report.json').read_text());assert n['native_bits_equal']and n['conditions']==300
 for path,h in n['source_hashes'].items():assert sha(ROOT/path)==h
 for path,h in json.loads((D/'workflow-hashes.json').read_text()).items():assert sha(ROOT/path)==h
 files=list(dict.fromkeys([Path(__file__),D/'build/report.json',D/'workflow-hashes.json',D/'frozen-builder.py',D/'native-check/report.json']+[ROOT/p for p in r['source_hashes']]+[ROOT/p for p in n['source_hashes']]+[ROOT/p for p in json.loads((D/'workflow-hashes.json').read_text())]))
 result=dict(module=r['wasm_sha256'],all_original_22_wat_sources_identical=True,all_other_runtime_rs_identical=unchanged,wrapper_rs_byte_exact=True,replacement_source_byte_exact=True,native_conditions=300,native_bits_equal=True,wasm_full_inference_fidelity_verified=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (D/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(D/'source-audit.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'source-audit.json']:z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(D/'source-audit.zip')as z:
  assert len(z.namelist())==len(set(z.namelist()))
  for p,h in result['source_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h
 print(json.dumps(dict(module=r['wasm_sha256'],original22_identical=True,native_conditions=300,unique_archive_verified=True)))
if __name__=='__main__':main()
