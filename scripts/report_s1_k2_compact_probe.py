#!/usr/bin/env python3
"""Saved Candid and canonical rank7 integer audit."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-k2-compact-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/report_s2_pair_late_probe.py';s=p.read_text().replace('s2-pair-late-v1','s1-k2-compact-v1').replace('check-v2/report.json','check/report.json').replace('s2_pair_late','winograd7').replace("len(r['cases']) == 19","len(r['cases']) == 21").replace('ordinary_queries=38','ordinary_queries=42')
 (D/'frozen-reporter.py').write_text(s);exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 r=json.loads((D/'summary.json').read_text());build=json.loads((D/'build/report.json').read_text());check=json.loads((D/'check/report.json').read_text())
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in build[key].items())
 assert r['module']==build['wasm_sha256']==sha(D/'build/diagnostic.wasm') and check['ordinary_queries']==42
 ns={'__name__':'rank7','__file__':str(D/'plan.py')};exec(compile((D/'plan.py').read_text(),str(D/'plan.py'),'exec'),ns)
 a,b,c,leaves,roots,_=ns['plan']();proof=json.loads((D/'integer-bounds.json').read_text());bounds=[]
 for name,e in c.symbols.items():
  terms={}
  for m,sign in e.items():
   for ai,av in a.symbols[leaves[m][0]].items():
    for bi,bv in b.symbols[leaves[m][1]].items():terms[ai,bi]=terms.get((ai,bi),0)+sign*av*bv
  bound=sum(abs(v)for v in terms.values())*127*128*128;assert bound<2**31;bounds.append(dict(name=name,bound=bound))
 assert proof['bounds']==bounds and proof['input_i16_bound']==508 and proof['weight_i16_bound']==512
 for case in check['cases']:assert sha(D/'check'/f"{case['label']}.input.bin")==case['input_sha256']
 files=[Path(__file__),ROOT/'scripts/build_s1_winograd_probe.py',ROOT/'scripts/check_s1_winograd_probe.py',p,D/'plan.py',D/'integer-bounds.json',ROOT/'scripts/check_s1_k2_compact_probe.py',D/'frozen-reporter.py',D/'build/report.json',D/'check/report.json']+[ROOT/p for p in build['source_hashes']]+list((D/'check').glob('*.hex'));files=list(dict.fromkeys(files))
 r.update(integer_bounds_verified=True,rank=7,output_tile=128,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Same raw weight capacity and values with audited K2/four-column layout, unchanged quantization/scales and F32 block order; includes input preparation. Diagnostic Q projection only.');(D/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
if __name__=='__main__':main()
