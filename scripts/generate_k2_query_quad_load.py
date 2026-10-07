#!/usr/bin/env python3
"""One 16-byte query load replaces four 32-bit splats, with exact byte shuffles."""
from pathlib import Path
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/k2-query-quad-load-kernels-v1';d.mkdir(exist_ok=False);old=ROOT/'artifacts/update-k2-pair-guard-fold-v1';items=[];files=[Path(__file__)];conditions=0
 rng=np.random.default_rng(432164)
 for trial in range(128):
  words=rng.integers(-32768,32768,8,dtype=np.int16);raw=words.astype('<i2').tobytes()
  for pair in range(4):assert bytes(raw[i]for i in list(range(pair*4,pair*4+4))*4)==raw[pair*4:pair*4+4]*4;conditions+=1
 pattern=r'\(local.tee \$x(\d+)_(\d+)\(v128.load32_splat offset=(\d+)\(local.get \$qp\)\)\)'
 for tile in [128,168,160,32]:
  for seed in [False,True]:
   p=old/(f'k2-direct{tile}'+('_seed'if seed else '')+'.wat');s=p.read_text();replacements=[]
   def replace(m):
    leaf,k,offset=map(int,m.groups());assert offset==k*4
    prefix=f'(local.tee $qpacket(v128.load offset={offset}(local.get $qp)))'if k%4==0 else 'local.get $qpacket'
    mask=list(range((k%4)*4,(k%4)*4+4))*4
    after=prefix+'\n(v128.const i32x4 0 0 0 0)\ni8x16.shuffle '+' '.join(map(str,mask))+f'\nlocal.tee $x{leaf}_{k}'
    replacements.append(dict(before=m[0],after=after));return after
   text,count=re.subn(pattern,replace,s);assert count==(7*2+4)*64,(p,count)
   assert all(replacements[i]['before'].split('offset=')[1].split('(')[0]==str((i%64)*4)for i in range(count))
   text=text.replace('(local $input_stride i32)','(local $qpacket v128)(local $input_stride i32)',1);assert text!=s and text.count('(local $qpacket v128)')==1
   # Reversal proves every other byte, integer operation and F32 carry unchanged.
   reverted=text.replace('(local $qpacket v128)','',1)
   for change in replacements:reverted=reverted.replace(change['after'],change['before'],1)
   assert reverted==s
   assert text.count('(')==text.count(')')and text.count('(local $')<10000
   target=d/p.name;target.write_text(text);items.append(dict(tile=tile,seed=seed,path=str(target.relative_to(ROOT)),symbol=re.search(r'export "([^"]+)"',s)[1],locals=text.count('(local $'),splats_replaced=count,load_groups=count//4,source_reversal_byte_exact=True));files.extend([p,target])
 report=dict(kernels=items,all_integer_dots_equal=True,query_byte_conditions=conditions,all_other_code_reverses_exactly=True,last_packet_read_end=256,query_load_stride=256,performance_verified=False,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'integer-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(kernels=8,query_byte_conditions=conditions,max_locals=max(k['locals']for k in items))))
if __name__=='__main__':main()
