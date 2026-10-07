#!/usr/bin/env python3
"""Re-decode adaptive168 measurements and repeat independent native digests."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-stack-store-all-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 upstream=ROOT/'scripts/report_s2_pair_late_probe.py'
 s=upstream.read_text().replace('s2-pair-late-v1','s1-stack-store-all-v1').replace('check-v2/report.json','check/report.json').replace('s1_pair_bounds','adaptive160').replace('s2_pair_late','stack_store').replace("len(r['cases']) == 19","len(r['cases']) == 21").replace('ordinary_queries=38','ordinary_queries=42')
 (D/'frozen-reporter.py').write_text(s)
 exec(compile(s,str(upstream),'exec'),dict(__file__=__file__,__name__='__main__'))
 b=json.loads((D/'build/report.json').read_text());c=json.loads((D/'check/report.json').read_text());r=json.loads((D/'summary.json').read_text())
 for hashes in [b['source_hashes'],b['dependency_hashes']]:
  assert all(sha(ROOT/p)==h for p,h in hashes.items())
 import re
 locals_count=0
 pairs=[(ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat',D/'build/kernel-store-0.wat'),(ROOT/'artifacts/s1-adaptive160-v1/build/kernel.wat',D/'build/kernel-store-1.wat'),(ROOT/'artifacts/s1-adaptive160-v1/build/kernel32.wat',D/'build/kernel-store-2.wat')]
 ops=lambda text:re.findall(r'(?m)^(?:f32x4\.[a-z0-9_]+|i32x4\.[a-z0-9_]+|i16x8\.[a-z0-9_]+|i8x16\.shuffle[^\n]*)$',text)
 for old,new in pairs:
  text=new.read_text();count=text.count('(local ');assert count<10000;locals_count=max(locals_count,count);assert ops(old.read_text())==ops(text)
 assert sum(b['store_sites'].values())==400
 assert b['wasm_sha256']==sha(D/'build/diagnostic.wasm')==r['module']
 assert all(p['wasmparser_validation'] for p in b['patches'])
 assert sha(ROOT/'scripts/check_s1_output160.py')==(D/'check-upstream.sha256').read_text().strip()
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
 for case in c['cases']:
  inp=D/'check'/f"{case['label']}.input.bin";assert sha(inp)==case['input_sha256']
  weights=D/'check'/f"weights-{case['rows']}-native.bin"
  native=json.loads(subprocess.check_output([str(helper),'native',str(case['tokens']),str(case['rows']),str(case['cols']),str(weights),str(inp)],text=True))
  assert native==case['native'],case['label']
 files=[Path(__file__),upstream,ROOT/'scripts/build_s1_adaptive168_probe.py',ROOT/'scripts/check_s1_adaptive168_probe.py',D/'build/report.json',D/'check/report.json',D/'frozen-check.py',D/'check-upstream.sha256',D/'frozen-reporter.py']+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'))
 files=list(dict.fromkeys(files))
 r.update(native_digests_recomputed=True,local_limit_verified=True,locals=locals_count,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Diagnostic Q projection only, same adaptive160/128/32 in both branches. Only400 output-store sites changed, 2 instructions saved/site; integer and F32 arithmetic opcode order unchanged. Full inference unmeasured; not adopted.')
 (D/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(complete=True,cases=len(c['cases']),native_digests_recomputed=True,locals=locals_count)))
if __name__=='__main__':main()
