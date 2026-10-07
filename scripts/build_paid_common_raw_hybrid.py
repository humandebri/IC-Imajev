#!/usr/bin/env python3
"""Use unchanged paid scheduler/contract with the full common-layout hybrid."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/paid-k2-pair-guard-fold-v1';normal=ROOT/'artifacts/update-common-raw-hybrid-v1';d=ROOT/'artifacts/paid-common-raw-hybrid-v1';d.mkdir(exist_ok=False)
 r=json.loads((normal/'build/report.json').read_text());assert len(r['patches'])==34
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in r[key].items())
 (d/'optimized-paid-scheduler.rs').write_bytes((old/'optimized-paid-scheduler.rs').read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/paid-k2-pair-guard-fold-v1','artifacts/paid-common-raw-hybrid-v1').replace('artifacts/update-k2-pair-guard-fold-v1','artifacts/update-common-raw-hybrid-v1')
 a=s.index('  paths=list(B.parent.glob(');b=s.index('\n  assert sha(wat)',a)
 s=s[:a]+"  mapping={sha(ROOT/p):ROOT/p for p in base['source_hashes'] if p.endswith('.wat')};wat=mapping[patch['source_sha256']]"+s[b:]
 s=s.replace("assert len({p['function_index'] for p in patches})==30","assert len({p['function_index'] for p in patches})==34")
 s=s.replace("baseline=base['baseline']","baseline='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'")
 frozen=d/'frozen-builder.py';frozen.write_text(s)
 files=[Path(__file__),old/'frozen-builder.py',old/'optimized-paid-scheduler.rs',normal/'build/report.json',frozen,d/'optimized-paid-scheduler.rs'];(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
