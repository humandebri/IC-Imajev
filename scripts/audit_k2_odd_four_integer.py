#!/usr/bin/env python3
"""Independent original dot, cached B recovery and emitted tail instruction audit."""
from pathlib import Path
import json,hashlib,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/k2-odd-four-kernels-v1';r=json.loads((d/'report.json').read_text());rng=np.random.default_rng(74004);conditions=0
 for pattern in ['random','minmax','zero','alternating']:
  for groups in [1,4,16,21]:
   for trial in range(12):
    a=rng.integers(-127,128,(2,128),dtype=np.int16);b=rng.integers(-128,128,(4,groups,4,128),dtype=np.int16)
    if pattern!='random':
     values=np.array([0]if pattern=='zero'else [-128,127]if pattern=='minmax'else [-127,0,127,1,-1],dtype=np.int16);b=np.resize(values,b.shape);a=np.resize(np.clip(values,-127,127),a.shape)
    # Cached operands in completed kernels: w0=B00,w1=B10,w2=B11,w4=B01-B00.
    w0,w1,w2,w4=b[0],b[1],b[3],b[2]-b[0];assert np.array_equal(w0+w4,b[2]);assert np.max(np.abs(w4))<=255
    pc0=(a[0].astype(np.int64)*w0).sum(axis=-1);pc1=(a[1].astype(np.int64)*w1).sum(axis=-1);pc2=(a[0].astype(np.int64)*(w0+w4)).sum(axis=-1);pc3=(a[1].astype(np.int64)*w2).sum(axis=-1)
    got=np.stack([pc0+pc1,pc2+pc3],axis=-1).reshape(groups,8)
    expected=np.stack([(a[0].astype(np.int64)*b[0]+a[1].astype(np.int64)*b[1]).sum(axis=-1),(a[0].astype(np.int64)*b[2]+a[1].astype(np.int64)*b[3]).sum(axis=-1)],axis=-1).reshape(groups,8);assert np.array_equal(got,expected);assert np.max(np.abs(got))<=256*127*128;conditions+=1
 for item in r['kernels']:
  p=ROOT/item['path'];text=p.read_text();old=(ROOT/'artifacts/update-k2-compact-v1'/p.name).read_text();anchor=')(else\n(local.set $qo';lo=text.index(anchor)+len(')(else\n');end=text.index('(local.set $t(i32.add(local.get $t)(i32.const 2)))',lo);olo=old.index(anchor)+len(')(else\n');oe=old.index('(local.set $t(i32.add(local.get $t)(i32.const 2)))',olo)
  assert text[:lo]==old[:olo] and text[end:]==old[oe:];tail=text[lo:end]
  assert set(re.findall(r'local.get \$x(\d+)_',tail))=={'0','1'}
  assert set(re.findall(r'local.get \$w(\d+)_',tail))=={'0','1','2','4'}
  assert tail.count('i16x8.add')==64*(item['tile']//8)
  for j in range(item['tile']//8):
   for k in range(64):assert f'local.get $w0_{j}_{k}\nlocal.get $w4_{j}_{k}\ni16x8.add\ni32x4.dot_i16x8_s' in tail
  assert tail.count('f32x4.convert_i32x4_s')==item['tile']//4
 files=[Path(__file__),d/'report.json']+[ROOT/p for p in r['source_hashes']]
 result=dict(complete=True,conditions=conditions,all_integer_dots_equal=True,emitted_cached_weight_recovery_verified=True,first_full_pair_and_loop_suffix_byte_equal=True,original_dot_bound=256*127*128,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},performance_verified=False)
 (d/'integer-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='source_hashes'}))
if __name__=='__main__':main()
