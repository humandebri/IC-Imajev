#!/usr/bin/env python3
"""Verify full-quartet paths and diagnostic/control implementation against alias baseline."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-tail-dispatch-probe-v1';k=ROOT/'artifacts/s2-k2-tail-dispatch-kernels-v1';old=ROOT/'artifacts/s2-k2-alias-kernels-v1';r=json.loads((k/'report.json').read_text());files=[]
 for item in r['kernels']:
  p=ROOT/item['path'];lines=p.read_text().splitlines();before=(old/p.name).read_text().splitlines()
  for loop in [False,True]:
   lo=lines.index('(loop $tokens')if loop else 0;start=next(i for i in range(lo,len(lines))if lines[i].startswith('(local.set $qoff('));end=next(i for i in range(start,len(lines))if lines[i]==')(else')
   olo=before.index('(loop $tokens')if loop else 0;os=next(i for i in range(olo,len(before))if before[i].startswith('(local.set $qoff('));oe=next(i for i in range(os,len(before))if before[i].startswith('(local.set $t('))
   assert lines[start:end]==before[os:oe]
  files.extend([p,old/p.name])
 b=json.loads((d/'build/report.json').read_text());prev=ROOT/'artifacts/s2-k2-alias-probe-v1';pb=json.loads((prev/'build/report.json').read_text());assert len(b['patches'])==34
 assert [x['source_sha256']for x in b['patches'][:30]]==[x['source_sha256']for x in pb['patches'][:30]]
 for p in (d/'build/src').glob('*.rs'):assert p.read_bytes()==(prev/'build/src'/p.name).read_bytes();files.extend([p,prev/'build/src'/p.name])
 files.extend([Path(__file__),d/'build/report.json',k/'layout-audit.json'])
 result=dict(complete=True,full_quartet_path_byte_equal=True,all_diagnostic_rust_byte_equal=True,all30_control_wat_byte_equal=True,adopted=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({x:y for x,y in result.items()if x!='source_hashes'}))
if __name__=='__main__':main()
