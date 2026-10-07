#!/usr/bin/env python3
"""Permute raw output lanes and remove only four output shuffle patterns."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/k2-contiguous-roots-kernels-v1';d.mkdir(exist_ok=False);files=[Path(__file__)];kernels=[]
 for tile in [128,168,160,32]:
  for seed in [False,True]:
   p=ROOT/'artifacts/update-k2-pair-guard-fold-v1'/f'k2-direct{tile}{"_seed" if seed else ""}.wat';s=p.read_text();count=0
   for a,b in [(7,10),(11,12)]:
    for mask,root in [('0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23',a),('8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31',b)]:
     pattern=rf'local.get \$pc{a}\s+local.get \$pc{b}\s+i8x16.shuffle {mask}(?=\s)';s,n=re.subn(pattern,f'local.get $pc{root}',s);count+=n
   assert count==tile//8*10 and 'i8x16.shuffle'not in s,(tile,count)
   out=d/p.name;out.write_text(s);files.extend([p,out]);kernels.append(dict(path=str(out.relative_to(ROOT)),tile=tile,seed=seed,symbol=re.search(r'export "([^"]+)"',s)[1],removed_shuffles=count))
 r=dict(complete=True,kernels=kernels,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Raw four-plane layout permutation; only output shuffle/get pairs removed. Requires new raw producer. No performance claim.');(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(kernels=len(kernels),removed_shuffles=sum(k['removed_shuffles']for k in kernels))))
if __name__=='__main__':main()
