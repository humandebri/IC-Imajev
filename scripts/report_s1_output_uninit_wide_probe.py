#!/usr/bin/env python3
"""Re-decode adaptive168 measurements and repeat independent native digests."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-output-uninit-wide-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'scripts/report_s2_pair_late_probe.py'
 s=upstream.read_text().replace('s2-pair-late-v1','s1-output-uninit-wide-v1').replace('check-v2/report.json','check/report.json').replace('s1_pair_bounds','direct_output').replace('s2_pair_late','output_uninit').replace("len(r['cases']) == 19","len(r['cases']) == 21").replace('ordinary_queries=38','ordinary_queries=42')
 (D/'frozen-reporter.py').write_text(s)
 exec(compile(s,str(upstream),'exec'),dict(__file__=__file__,__name__='__main__'))
 b=json.loads((D/'build/report.json').read_text());c=json.loads((D/'check/report.json').read_text());r=json.loads((D/'summary.json').read_text())
 for hashes in [b['source_hashes'],b['dependency_hashes']]:
  assert all(sha(ROOT/p)==h for p,h in hashes.items())
 old=json.loads((ROOT/'artifacts/s1-direct-output-v1/build/report.json').read_text())
 assert [p['source_sha256']for p in b['patches']]==[p['source_sha256']for p in old['patches']]
 assert len({p['function_index'] for p in b['patches']})==len(b['patches'])==15
 locals_count=9504
 exact=(D/'build/src/exact.rs').read_text();body=exact[exact.index('pub fn project_uninit('):exact.index('pub fn project_direct(')]
 assert 'MaybeUninit<f32>' in body and 'Vec::from_raw_parts' in body
 assert 'out[t*rows' not in body and 'out.as_mut_ptr().add' not in body
 assert 'self.cols<256||self.cols%256!=0' in body
 coverage=[]
 for rows in [32,64,128,160,256,512,1280,2560,4096,8192,9216]:
  ranges=[];start=0
  while start<rows:
   tile=160 if rows-start>=160 else 128 if rows-start>=128 else 32
   width=min(tile,rows-start);ranges.append((start,start+width));start+=width
  for n in range(1,133):
   if n*rows>900000:continue
   tokens=[0]+([1]if n>1 else[]);t=2
   while t<(n&~1):tokens.extend([t,t+1]);t+=2
   if t<n:tokens.append(t)
   assert tokens==list(range(n));end=0
   for t in tokens:
    for lo,hi in ranges:assert t*rows+lo==end;end=t*rows+hi
   assert end==n*rows;coverage.append(dict(rows=rows,tokens=n,output_values=end,contiguous_writes_once=True))
 coverage_path=D/'output-address-coverage.json';coverage_path.write_text(json.dumps(dict(complete=True,cases=coverage),indent=2)+'\n')
 assert b['wasm_sha256']==sha(D/'build/diagnostic.wasm')==r['module']
 assert all(p['wasmparser_validation'] for p in b['patches'])
 assert sha(ROOT/'scripts/check_s1_output160.py')==(D/'check-upstream.sha256').read_text().strip()
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
 for case in c['cases']:
  inp=D/'check'/f"{case['label']}.input.bin";assert sha(inp)==case['input_sha256']
  weights=D/'check'/f"weights-{case['rows']}-native.bin"
  native=json.loads(subprocess.check_output([str(helper),'native',str(case['tokens']),str(case['rows']),str(case['cols']),str(weights),str(inp)],text=True))
  assert native==case['native'],case['label']
 files=[coverage_path,Path(__file__),upstream,ROOT/'scripts/build_s1_output_uninit_wide_probe.py',ROOT/'scripts/check_s1_output_uninit_wide_probe.py',D/'build/report.json',D/'check/report.json',D/'frozen-check.py',D/'check-upstream.sha256',D/'frozen-reporter.py']+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'))
 files=list(dict.fromkeys(files))
 r.update(native_digests_recomputed=True,local_limit_verified=True,locals=locals_count,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Diagnostic Q projection only. Same15 kernels; final output zero-fill removed. Uninitialized allocation accessed only via raw pointer writes until all output is initialized. Full inference unmeasured; not adopted.')
 (D/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,cases=len(c['cases']),native_digests_recomputed=True,locals=locals_count)))
if __name__=='__main__':main()
