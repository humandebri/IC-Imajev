#!/usr/bin/env python3
"""Add a direct compact rank7 tile160 for actual MLP control comparisons."""
from pathlib import Path
import hashlib,json,textwrap
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s1-k2-single-v1';d=ROOT/'artifacts/s1-k2-mlp160-v1';d.mkdir(exist_ok=False)
 s=(old/'frozen-builder.py').read_text().replace('s1-k2-single-v1','s1-k2-mlp160-v1')
 generator=(ROOT/'scripts/build_s1_k2_four_columns_adaptive168_probe.py').read_text()
 function=textwrap.dedent(generator[generator.index(' def make_kernel(TILE):'):generator.index(' kernels=[make_kernel')])
 (d/'kernel160-generator.py').write_text(function)
 patch='''
 namespace=dict(runpy=__import__('runpy'),d=d,b=b,weight=__import__('build_s1_winograd_fused_probe').weight)
 exec(compile((d/'kernel160-generator.py').read_text(),str(d/'kernel160-generator.py'),'exec'),namespace)
 wat,count=namespace['make_kernel'](160);assert count<10000
 p=src/'win_kernel.rs';text=p.read_text()
 for seed in (False,True):
  symbol='__imajev_win7_wide_accumulate'+('_seed'if seed else '')
  fn='wide'+('_seed'if seed else '')
  lo=text.index('#[export_name="'+symbol+'"]');hi=text.index('\\n}',lo)+2
  text+='\\n'+text[lo:hi].replace(symbol,symbol.replace('wide','160')).replace('fn '+fn+'(','fn wide160'+('_seed'if seed else '')+'(').replace('n*W','n*160')
 p.write_text(text)
 p=src/'winograd.rs';text=p.read_text()
 before='else if rows-r>=128{128}';assert text.count(before)==1;text=text.replace(before,'else if rows!=8192&&rows!=4096&&rows-r>=160{160}'+before)
 before='match (tile,b==0){';assert text.count(before)==1;text=text.replace(before,before+'(160,true)=>crate::win_kernel::wide160_seed,(160,false)=>crate::win_kernel::wide160,')
 p.write_text(text)
 for seed in (False,True):
  symbol='__imajev_win7_160_accumulate'+('_seed'if seed else '')
  text=__import__('build_s1_k2_direct_probe').direct(wat.read_text(),160,seed).replace('__imajev_win7_wide_accumulate',symbol)
  before='(local.set $qo(i32.shl(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start))(i32.const 1)))'
  after='(local.set $qo(i32.add(i32.mul(i32.shr_u(local.get $t)(i32.const 1))(local.get $cols))(local.get $start)))'
  assert text.count(before)==3;text=text.replace(before,after)
  target=b/('direct160'+('_seed'if seed else '')+'.wat');target.write_text(text);generated.append((target,symbol))
'''
 anchor=" p=src/'lib.rs'\n cmd=nb['command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),old/'frozen-builder.py',ROOT/'scripts/build_s1_k2_four_columns_adaptive168_probe.py',ROOT/'scripts/build_s1_winograd_fused_probe.py',ROOT/'scripts/build_s1_k2_direct_probe.py',d/'kernel160-generator.py',d/'frozen-builder.py']
 (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
