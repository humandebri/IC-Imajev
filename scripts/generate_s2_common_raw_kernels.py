#!/usr/bin/env python3
"""Adapt contiguous rank49 addressing to shared four-plane raw bank."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/s2-contiguous-roots-kernels-v1';r=json.loads((old/'report.json').read_text());d=ROOT/'artifacts/s2-common-raw-kernels-v1';d.mkdir(exist_ok=True);files=[Path(__file__),old/'report.json'];kernels=[]
 for item in r['kernels']:
  p=ROOT/item['path'];s=p.read_text()
  for m in range(16):
   before=f'(local.set $wp{m}(i32.add(i32.load offset={m*4}(local.get $w))(local.get $start)))';assert s.count(before)==1;s=s.replace(before,before.replace('(local.get $start)','(i32.shl(local.get $start)(i32.const 1))'))
  pattern=r'(\(local.set \$bp\d+\(i32.add\(local.get \$wp\d+\)\(i32.mul\(local.get \$cols\)\(i32.const )(\d+)(\)\)\)\))';s,count=re.subn(pattern,lambda m:m[1]+str(int(m[2])*4)+m[3],s);assert count==item['tile']//16*16
  out=d/p.name;out.write_text(s);new=dict(item,path=str(out.relative_to(ROOT)),source_sha256=sha(out));kernels.append(new);files.extend([p,out])
 result=dict(complete=True,kernels=kernels,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Only W start-byte and group-stride addressing changed. Caller must supply 16 shared-bank aliases. No performance claim.');(d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(kernels=len(kernels))))
if __name__=='__main__':main()
