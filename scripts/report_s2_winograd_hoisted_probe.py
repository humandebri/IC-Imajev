#!/usr/bin/env python3
"""Re-decode all rank49 Winograd replies and independently audit canonical bounds."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s2-winograd-hoisted-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/report_s2_pair_late_probe.py';s=p.read_text().replace('s2-pair-late-v1','s2-winograd-hoisted-v1').replace('check-v2/report.json','check/report.json').replace('s2_pair_late','s2_winograd_hoisted').replace("len(r['cases']) == 19","len(r['cases']) == 21").replace('ordinary_queries=38','ordinary_queries=42')
 (D/'frozen-reporter.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 r=json.loads((D/'summary.json').read_text());b=json.loads((D/'build/report.json').read_text());q=json.loads((D/'check/report.json').read_text());assert r['module']==b['wasm_sha256']==sha(D/'build/diagnostic.wasm');assert q['ordinary_queries']==42
 for manifest in [b['source_hashes'],b['dependency_hashes'],json.loads((D/'entry-hashes.json').read_text())]:assert all(sha(ROOT/p)==h for p,h in manifest.items())
 ns=dict(__file__=str(D/'frozen-plan.py'),__name__='verify49');exec(compile((D/'frozen-plan.py').read_text(),'<verify49>','exec'),ns);a,bdag,c,leaves,roots,_=ns['plan']();bounds=json.loads((D/'integer-bounds.json').read_text());actual=[]
 for name,e in c.symbols.items():
  coefficients={}
  for m,sign in e.items():
   for ai,av in a.symbols[leaves[m][0]].items():
    for bi,bv in bdag.symbols[leaves[m][1]].items():coefficients[ai,bi]=coefficients.get((ai,bi),0)+sign*av*bv
  bound=sum(abs(v)for v in coefficients.values())*127*128*64;assert bound<2**31;actual.append(dict(name=name,bound=bound))
 assert actual==bounds['bounds'] and len(c.nodes)==77 and len(leaves)==49
 assert bounds['input_i16_bound']==max(sum(abs(v)for v in e.values())for e in a.symbols.values())*127<32768
 assert bounds['weight_i16_bound']==max(sum(abs(v)for v in e.values())for e in bdag.symbols.values())*128<32768
 for case in q['cases']:assert sha(D/'check'/f"{case['label']}.input.bin")==case['input_sha256']
 previous=ROOT/'artifacts/s2-winograd-v1/summary.json';by={v['label']:v for v in json.loads(previous.read_text())['cases']}
 for case in r['cases']:
  if case['label']in by:
   old=by[case['label']];assert old['tokens']==case['tokens'];case['previous_instructions']=old['after'];case['versus_previous_percent']=100*(1-case['after']/old['after'])
 r.update(integer_bounds_verified=True,rank=49,output_tile=88,previous_summary_sha256=sha(previous),scope='Q projection only; all original token/weight/scale data and FP block order preserved. Exact rank49 Winograd DAG with padded weight tail; includes quantization/input preparation. Not connected to whole inference.')
 paths=[D/'source-audit.json',D/'upstream-hashes.json',Path(__file__),p,ROOT/'scripts/build_s2_winograd_hoisted_probe.py',ROOT/'scripts/check_s2_winograd_hoisted_probe.py',D/'frozen-plan.py',D/'frozen-builder.py',D/'frozen-check.py',D/'frozen-reporter.py',D/'integer-bounds.json',D/'entry-hashes.json',D/'build/report.json',D/'check/report.json',previous]+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'))
 paths=list(dict.fromkeys(paths));r['workflow_hashes']={str(p.relative_to(ROOT)):sha(p)for p in paths};(D/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print('21 cases /42 replies, symbolic identity and all integer bounds verified')
if __name__=='__main__':main()
