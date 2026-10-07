#!/usr/bin/env python3
"""Re-decode adaptive168 measurements and repeat independent native digests."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-direct-output-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'scripts/report_s2_pair_late_probe.py'
 s=upstream.read_text().replace('s2-pair-late-v1','s1-direct-output-v1').replace('check-v2/report.json','check/report.json').replace('s1_pair_bounds','zero_seed').replace('s2_pair_late','direct_output').replace("len(r['cases']) == 19","len(r['cases']) == 21").replace('ordinary_queries=38','ordinary_queries=42')
 (D/'frozen-reporter.py').write_text(s)
 exec(compile(s,str(upstream),'exec'),dict(__file__=__file__,__name__='__main__'))
 b=json.loads((D/'build/report.json').read_text());c=json.loads((D/'check/report.json').read_text());r=json.loads((D/'summary.json').read_text())
 for hashes in [b['source_hashes'],b['dependency_hashes']]:
  assert all(sha(ROOT/p)==h for p,h in hashes.items())
 import re
 locals_count=0
 pairs=[]
 for i in range(3):
  pairs.extend([(ROOT/f'artifacts/s1-stack-store-all-v1/build/kernel-store-{i}.wat',D/f'build/direct-{i}-__imajev_direct_.wat'),(ROOT/f'artifacts/s1-zero-seed-v1/build/kernel-seed-{i}.wat',D/f'build/direct-{i}-__imajev_directseed_.wat')])
 ops=lambda text:re.findall(r'(?m)^(?:f32x4\.[a-z0-9_]+|i32x4\.[a-z0-9_]+|i16x8\.[a-z0-9_]+|i8x16\.shuffle[^\n]*)$',text)
 for old,new in pairs:
  text=new.read_text();count=text.count('(local ');assert count<10000;locals_count=max(locals_count,count);assert ops(old.read_text())==ops(text)
 assert sum(b['row1_store_sites'].values())==320
 exact=(D/'build/src/exact.rs').read_text();body=exact[exact.index('pub fn project_direct('):exact.index('pub fn project_seed(')];assert 'let mut sums' not in body and 'copy_from_slice(&sums' not in body and 'assert_eq!(width,tile)' in body
 coverage=[]
 for rows in [32,64,128,160,256,512,1280,2560,4096,8192,9216]:
  ranges=[];block_start=0
  while block_start<rows:
   tile=160 if rows-block_start>=160 else 128 if rows-block_start>=128 else 32
   assert block_start+tile<=rows;ranges.append((block_start,block_start+tile));block_start+=tile
  for n in range(1,133):
   if n*rows>900000:continue
   tokens=[0]+([1]if n>1 else[]);t=2
   while t<(n&~1):tokens.extend([t,t+1]);t+=2
   if t<n:tokens.append(t)
   assert tokens==list(range(n));end=0
   for t in tokens:
    for lo,hi in ranges:
     assert t*rows+lo==end;end=t*rows+hi
   assert end==n*rows
   coverage.append(dict(rows=rows,tokens=n,output_values=end,contiguous_writes_once=True))
 coverage_path=D/'output-address-coverage.json';coverage_path.write_text(json.dumps(dict(complete=True,cases=coverage,scope='Interval coverage for adaptive160/128/32, dynamic output stride and pair/odd token loop. Arithmetic verified separately by Wasm/native digests.'),indent=2)+'\n')
 assert b['wasm_sha256']==sha(D/'build/diagnostic.wasm')==r['module']
 assert all(p['wasmparser_validation'] for p in b['patches'])
 assert sha(ROOT/'scripts/check_s1_output160.py')==(D/'check-upstream.sha256').read_text().strip()
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
 for case in c['cases']:
  inp=D/'check'/f"{case['label']}.input.bin";assert sha(inp)==case['input_sha256']
  weights=D/'check'/f"weights-{case['rows']}-native.bin"
  native=json.loads(subprocess.check_output([str(helper),'native',str(case['tokens']),str(case['rows']),str(case['cols']),str(weights),str(inp)],text=True))
  assert native==case['native'],case['label']
 files=[coverage_path,Path(__file__),upstream,ROOT/'scripts/build_s1_adaptive168_probe.py',ROOT/'scripts/check_s1_adaptive168_probe.py',D/'build/report.json',D/'check/report.json',D/'frozen-check.py',D/'check-upstream.sha256',D/'frozen-reporter.py']+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'))
 files=list(dict.fromkeys(files))
 r.update(native_digests_recomputed=True,local_limit_verified=True,locals=locals_count,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Diagnostic Q projection only, same zero-seed adaptive160/128/32 in both branches. Direct strided output eliminates tile buffer and copy. Positive-zero add retained. Integer and F32 arithmetic opcode order unchanged. Full inference unmeasured; not adopted.')
 (D/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,cases=len(c['cases']),native_digests_recomputed=True,locals=locals_count)))
if __name__=='__main__':main()
