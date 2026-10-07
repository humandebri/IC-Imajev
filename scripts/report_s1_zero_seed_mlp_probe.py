#!/usr/bin/env python3
"""Re-decode adaptive168 measurements and repeat independent native digests."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-zero-seed-mlp-v1';B=ROOT/'artifacts/s1-zero-seed-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'scripts/report_s2_pair_late_probe.py'
 s=upstream.read_text().replace('s2-pair-late-v1','s1-zero-seed-mlp-v1').replace('check-v2/report.json','check/report.json').replace('s1_pair_bounds','stack_store').replace('s2_pair_late','zero_seed').replace("len(r['cases']) == 19","len(r['cases']) == 16").replace('ordinary_queries=38','ordinary_queries=32')
 (D/'frozen-reporter.py').write_text(s)
 exec(compile(s,str(upstream),'exec'),dict(__file__=__file__,__name__='__main__'))
 b=json.loads((B/'report.json').read_text());c=json.loads((D/'check/report.json').read_text());r=json.loads((D/'summary.json').read_text())
 for hashes in [b['source_hashes'],b['dependency_hashes']]:
  assert all(sha(ROOT/p)==h for p,h in hashes.items())
 import re
 locals_count=0
 pairs=[(ROOT/f'artifacts/s1-stack-store-all-v1/build/kernel-store-{i}.wat',B/f'kernel-seed-{i}.wat')for i in range(3)]
 ops=lambda text:re.findall(r'(?m)^(?:f32x4\.[a-z0-9_]+|i32x4\.[a-z0-9_]+|i16x8\.[a-z0-9_]+|i8x16\.shuffle[^\n]*)$',text)
 for old,new in pairs:
  text=new.read_text();count=text.count('(local ');assert count<10000;locals_count=max(locals_count,count);assert ops(old.read_text())==ops(text)
  assert 'local.get $yp\nv128.load' not in text
 assert sum(b['initialized_store_sites'].values())==400
 exact=(B/'src/exact.rs').read_text();assert 'MaybeUninit<f32>' in exact and 'ManuallyDrop::new(sums)' in exact and 'Vec::from_raw_parts' in exact
 coverage=[]
 for tile in [32,128,160]:
  for n in range(1,133):
   tokens=[0]+([1]if n>1 else[]);t=2;fulln=n&~1
   while t<fulln:tokens.extend([t,t+1]);t+=2
   if t<n:tokens.append(t)
   assert sorted(tokens)==list(range(n)) and len(set(tokens))==n
   written=[0]*(n*tile)
   for t in tokens:
    for j in range(tile//4):
     for lane in range(4):written[t*tile+j*4+lane]+=1
   assert all(v==1 for v in written)
   coverage.append(dict(tile=tile,tokens=n,initialized_lanes=n*tile,each_lane_written_once=True))
 coverage_path=D/'initialization-coverage.json';coverage_path.write_text(json.dumps(dict(complete=True,cases=coverage,scope='Address coverage of fixed pair/odd token loops and4-lane stores. Actual Wasm arithmetic verified separately by native digest comparison.'),indent=2)+'\n')
 assert b['wasm_sha256']==sha(B/'diagnostic.wasm')==r['module']
 assert all(p['wasmparser_validation'] for p in b['patches'])
 assert sha(ROOT/'scripts/check_s1_output160.py')==(D/'check-upstream.sha256').read_text().strip()
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
 for case in c['cases']:
  inp=D/'check'/f"{case['label']}.input.bin";assert sha(inp)==case['input_sha256']
  weights=D/'check'/f"weights-{case['rows']}-native.bin"
  native=json.loads(subprocess.check_output([str(helper),'native',str(case['tokens']),str(case['rows']),str(case['cols']),str(weights),str(inp)],text=True))
  assert native==case['native'],case['label']
 files=[coverage_path,Path(__file__),upstream,ROOT/'scripts/build_s1_adaptive168_probe.py',ROOT/'scripts/check_s1_adaptive168_probe.py',B/'report.json',D/'check/report.json',D/'frozen-check.py',D/'check-upstream.sha256',D/'frozen-reporter.py']+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'))
 files=list(dict.fromkeys(files))
 r.update(native_digests_recomputed=True,local_limit_verified=True,locals=locals_count,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Diagnostic real layer3 MLP gate9216x2560 projection with Q-source activation size probes and synthetic boundaries, same stack-store adaptive160/128/32 in both branches. Every output initialized by block0 with positive-zero F32 add; MaybeUninit avoids redundant zero fill. Integer and F32 arithmetic opcode order unchanged. Full inference unmeasured; not adopted.')
 (D/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,cases=len(c['cases']),native_digests_recomputed=True,locals=locals_count)))
if __name__=='__main__':main()
