#!/usr/bin/env python3
"""Verify tail guards are the only changes to the completed alias kernels."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-tail-probe-v1';k=ROOT/'artifacts/s2-k2-tail-kernels-v1';old=ROOT/'artifacts/s2-k2-alias-kernels-v1';r=json.loads((k/'report.json').read_text());files=[];guards=0
 for item in r['kernels']:
  p=ROOT/item['path'];lines=p.read_text().splitlines();filtered=[];opens=closes=0
  for line in lines:
   if re.fullmatch(r'\(if\(i32.gt_u\(i32.sub\(local.get \$n\)\(local.get \$t\)\)\(i32.const \d+\)\)\(then',line):opens+=1;continue
   if re.fullmatch(r'\)\(else\(local.set \$pc\d+\(v128.const i32x4 0 0 0 0\)\)\)\)',line):closes+=1;continue
   filtered.append(line)
  assert opens==closes and opens>0;guards+=opens;assert '\n'.join(filtered)+'\n'==(old/p.name).read_text();files.extend([p,old/p.name])
 b=json.loads((d/'build/report.json').read_text());prev=ROOT/'artifacts/s2-k2-alias-probe-v1';pb=json.loads((prev/'build/report.json').read_text())
 assert len(b['patches'])==34
 assert [x['source_sha256']for x in b['patches'][:30]]==[x['source_sha256']for x in pb['patches'][:30]]
 for p in (d/'build/src').glob('*.rs'):assert p.read_bytes()==(prev/'build/src'/p.name).read_bytes();files.extend([p,prev/'build/src'/p.name])
 files.extend([Path(__file__),d/'build/report.json',k/'layout-audit.json'])
 result=dict(complete=True,guards=guards,removing_guards_restores_previous_kernel_byte_exact=True,all_diagnostic_rust_byte_equal=True,all30_control_wat_byte_equal=True,adopted=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (d/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({x:y for x,y in result.items()if x!='source_hashes'}))
if __name__=='__main__':main()
