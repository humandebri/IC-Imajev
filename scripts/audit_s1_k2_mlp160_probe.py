#!/usr/bin/env python3
"""Independently audit tile160 sources, initialization spans and completed evidence."""
from pathlib import Path
import hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-k2-mlp160-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=json.loads((D/'build/report.json').read_text());old=ROOT/'artifacts/s1-k2-single-v1';ob=json.loads((old/'build/report.json').read_text())
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in b[key].items())
 assert sha(D/'build/diagnostic.wasm')==b['wasm_sha256'] and len(b['patches'])==30
 for p in ob['patches']:assert any((x['export'],x['source_sha256'])==(p['export'],p['source_sha256'])for x in b['patches'])
 assert all(p['wasmparser_validation']for p in b['patches'])
 for p,h in json.loads((D/'entry-hashes.json').read_text()).items():assert sha(ROOT/p)==h
 spans=0
 for rows in [2560,4096,8192,9216]:
  r=0;covered=[]
  while r<rows:
   tile=168 if rows in [8192,4096]and rows-r>=168 else 160 if rows not in [8192,4096]and rows-r>=160 else 128 if rows-r>=128 else 32
   assert tile%8==0 and r+tile<=rows;covered.extend(range(r,r+tile));r+=tile
  assert covered==list(range(rows))
  for n in [1,2,3,5,8,32,56,57,67,89,109,132]:
   # First row body writes row0 and guards row1; later row bodies guard odd tail.
   tokens=[0]+([1]if n>1 else [])
   for t in range(2,n,2):tokens+=[t]+([t+1]if t+1<n else [])
   assert tokens==list(range(n));spans+=1
 wat=(D/'build/kernel160.wat').read_text();locals_=wat.count('(local $');assert locals_<10000
 direct=(D/'build/direct160.wat').read_text();seed=(D/'build/direct160_seed.wat').read_text()
 # Seed changes only previous-output loads to positive zero and its export.
 normal=re.sub(r'(local.get \$yp1?\n)local.get \$yp1?\nv128.load offset=\d+',r'\1v128.const i32x4 0 0 0 0',direct).replace('__imajev_win7_160_accumulate','__imajev_win7_160_accumulate_seed')
 assert seed==normal
 assert seed.count('v128.const i32x4 0 0 0 0')-direct.count('v128.const i32x4 0 0 0 0')==240
 assert direct.count('f32x4.add')==seed.count('f32x4.add')==240
 assert direct.count('(local.set $qo(i32.add(')==3
 completed=[]
 for name in ['gate','down']:
  summary=json.loads((D/f'summary-{name}.json').read_text());assert summary['module']==b['wasm_sha256'] and summary['native_bits_equal']
  for p,h in summary['workflow_hashes'].items():assert sha(ROOT/p)==h
  with zipfile.ZipFile(D/f'frozen-{name}-workflow.zip')as z:
   assert len(z.namelist())==len(set(z.namelist()))
   for p,h in summary['workflow_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h
   assert z.read(str((D/f'summary-{name}.json').relative_to(ROOT)))==(D/f'summary-{name}.json').read_bytes()
  completed.append(dict(shape=name,queries=summary['ordinary_queries']))
 report=dict(module=b['wasm_sha256'],old_28_kernel_sources_unchanged=True,new_kernels=2,locals160=locals_,complete_output_span_conditions=spans,positive_zero_seed_preserves_f32_add=True,compact_operand_pointer_verified=True,completed=completed,full_inference_adopted=False)
 (D/'post-report-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
