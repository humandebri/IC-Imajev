#!/usr/bin/env python3
"""Freeze author-provided numeric matrix data at one immutable Git commit; no code run."""
from pathlib import Path
import urllib.request,subprocess,json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/rank48-primary-coefficients-v1';d.mkdir(exist_ok=False)
 commit=subprocess.check_output(['git','ls-remote','https://github.com/jgdumas/plinopt.git','refs/heads/main'],text=True).split()[0];assert re.fullmatch('[0-9a-f]{40}',commit)
 names=['4x4x4_48_rational_'+x+'.sms'for x in ['L','R','P']]+['4x4x4_48_rational-'+kind+'_'+x+'.sms'for kind in ['ALT','CoB']for x in ['L','R','P']]+['Licence_CeCILL-B_V1-en.txt','AUTHORS.md'];files=[]
 for name in names:
  path=name if not name.endswith('.sms')else 'data/'+name;url='https://raw.githubusercontent.com/jgdumas/plinopt/'+commit+'/'+path
  with urllib.request.urlopen(url,timeout=30)as f:data=f.read(1_000_001)
  assert len(data)<1_000_000;p=d/name;p.write_bytes(data);files.append(dict(path=str(p.relative_to(ROOT)),url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
 r=dict(complete=True,commit=commit,repository='https://github.com/jgdumas/plinopt',paper='https://arxiv.org/abs/2506.13242',files=files,source_hashes={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},scope='Author coefficient data frozen as data only. No dependency installation, downloaded code execution, numerical or performance claim.');(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(commit=commit,files=len(files))))
if __name__=='__main__':main()
