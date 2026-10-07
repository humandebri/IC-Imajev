#!/usr/bin/env python3
"""Fetch pinned author SLP/matrix files as untrusted data; never execute programs."""
from pathlib import Path
import json,urllib.request,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/rank48-primary-coefficients-v1';r=json.loads((old/'report.json').read_text());tree=json.loads((old/'author-tree.json').read_text());paths=[v['path']for v in tree['tree']if v['type']=='blob'and v['path'].startswith('data/4x4x4_48_204')];d=ROOT/'artifacts/rank48-latest-programs-v1';d.mkdir(exist_ok=False);items=[]
 for path in paths:
  url='https://raw.githubusercontent.com/jgdumas/plinopt/'+r['commit']+'/'+path
  with urllib.request.urlopen(url,timeout=30)as f:data=f.read(1000001)
  assert len(data)<1000000;p=d/Path(path).name;p.write_bytes(data);items.append(dict(path=str(p.relative_to(ROOT)),url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
 files=[Path(__file__),old/'report.json',old/'author-tree.json',old/'author-tree-manifest.json'];report=dict(complete=True,source_commit=r['commit'],repository=r['repository'],files=items,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},scope='Pinned author numeric matrices and straight-line-program text. No code execution or dependency installation; no correctness/performance claim yet.');(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(files=len(items),commit=r['commit'])))
if __name__=='__main__':main()
