#!/usr/bin/env python3
"""Extend the exact bounded-pair S1 body to 160 outputs, sharing dead first-pair scratch."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-output160-v1/build'
def main():
 D.mkdir(parents=True,exist_ok=False);original=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat';s=original.read_text()
 # Scratch b12/b21 are consumed in the same output quartet where written.
 for plane in ['12','21']:
  for j in range(32):
   for k in range(32):
    name=f'$b{plane}_{j}_{k}';assert s.count('local.tee '+name+' ')==1 and s.count('local.get '+name+')')==0
    reads=list(re.finditer(r'local.get '+re.escape(name)+r'(?=[\s)])',s));assert len(reads)==1
    assert s.index('local.tee '+name+' ')<reads[0].start()
    if j<31:assert reads[0].start()<s.index(f'local.tee $b{plane}_{j+1}_{k} ')
 s=re.sub(r'\$b(12|21)_\d+_(\d+)\b',r'$b\1_\2',s);seen=set()
 def dedup(m):
  if m.group() in seen:return ''
  seen.add(m.group());return m.group()
 s=re.sub(r'\(local \$b(?:12|21)_\d+ v128\)',dedup,s)
 # Existing output row1 has byte stride512; row0 is unchanged.
 def stride(m):return m[1]+str(int(m[2])+128 if int(m[2])>=512 else int(m[2]))
 s=re.sub(r'(?m)^(v128\.(?:load|store) offset=)(\d+)$',stride,s)
 s=s.replace('(i32.shl (local.get $t) (i32.const 9))','(i32.mul (local.get $t) (i32.const 640))')
 declarations=[]
 for m in range(7):
  for j in range(32,40):
   for k in range(32):declarations.append(f'(local $w{m}_{j}_{k} v128)')
 declarations += [f'(local $sw{j} v128)' for j in range(32,40)]
 start=s.index('(local.set $sw0');s=s[:start]+'\n'.join(declarations)+'\n'+s[start:]
 start=s.index('(local.set $wp0');s=s[:start]+''.join(f'(local.set $sw{j} (v128.load offset={j*16} (local.get $sw)))\n' for j in range(32,40))+s[start:]
 def copy(block):
  block=re.sub(r'\$w([0-6])_(\d+)_(\d+)\b',lambda m:f'$w{m[1]}_{int(m[2])+24}_{m[3]}',block)
  block=re.sub(r'\$sw(\d+)\b',lambda m:f'$sw{int(m[1])+24}',block)
  block=re.sub(r'(?m)^(v128\.(?:load|store) offset=)(\d+)$',lambda m:m[1]+str(int(m[2])+384),block)
  block=re.sub(r'(\(i32.mul \(local.get \$cols\) \(i32.const )(\d+)(\)\))',lambda m:m[1]+str(int(m[2])+24)+m[3],block)
  return block
 # Clone quartets8..15. They reuse existing input locals and introduce no new input reads.
 row=lambda j:f'(local.set $row0 (i32.add (local.get $wp0) (i32.mul (local.get $cols) (i32.const {j}))))'
 lo=s.index(row(8));hi=s.index(row(16),lo);at=s.index('(local.set $t (i32.const 2))');s=s[:at]+copy(s[lo:hi])+s[at:]
 pair=s.index('(loop $tokens');end=s.index('(local.set $t (i32.add (local.get $t) (i32.const 2)))',pair)
 anchor=lambda j:f'local.get $x0_0\nlocal.get $w0_{j}_0'
 lo=s.index(anchor(8),pair);hi=s.index(anchor(16),lo);s=s[:end]+copy(s[lo:hi])+s[end:]
 tail=s.index('(if (i32.lt_u (local.get $t) (local.get $n)) (then');lo=s.index(anchor(8),tail);hi=s.index(anchor(16),lo)
 footer='\n))\n)\n))\n';assert s.endswith(footer);at=len(s)-len(footer);s=s[:at]+'\n'+copy(s[lo:hi])+s[at:]
 s=s.replace('__imajev_s1_wide_accumulate','__imajev_s1_160_accumulate');assert s.count('(')==s.count(')')
 count=len(re.findall(r'\(local \$',s));assert count<10000,count
 assert s.count('local.get $sw39')==5 # first two rows, full pair two rows, tail one.
 (D/'kernel.wat').write_text(s);sha=lambda b:hashlib.sha256(b).hexdigest()
 (D/'generator.json').write_text(json.dumps(dict(output_tile=160,locals=count,original_sha256=sha(original.read_bytes()),source_sha256=sha(Path(__file__).read_bytes()),wat_sha256=sha(s.encode()),fixed_weight_layout_unchanged=True,blocks_and_f32_order_unchanged=True,scratch_liveness_checked=True),indent=2)+'\n');print(count)
if __name__=='__main__':main()
