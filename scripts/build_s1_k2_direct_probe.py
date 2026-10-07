#!/usr/bin/env python3
"""Direct strided K2 output with block0 positive-zero add and no temporary tile copy."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def direct(text,tile,seed):
 text=text.replace('(local.get $stride)','(local.get $input_stride)')
 text=text.replace('(local $t i32)','(local $input_stride i32)(local $output_bytes i32)(local $t i32)',1)
 before='(local.set $sw0';at=text.index(before);text=text[:at]+'(local.set $input_stride(i32.shr_u(local.get $cols)(i32.const 8)))\n(local.set $output_bytes(i32.shl(local.get $stride)(i32.const 2)))\n'+text[at:]
 before=f'(i32.mul(local.get $t)(i32.const {tile*4}))';assert text.count(before)==3;text=text.replace(before,'(i32.mul(local.get $t)(local.get $output_bytes))')
 before=f'(i32.add(local.get $yp)(i32.const {tile*4}))';assert text.count(before)==3;text=text.replace(before,'(i32.add(local.get $yp)(local.get $output_bytes))')
 if seed:
  text,n=re.subn(r'(local.get \$yp1?\n)local.get \$yp1?\nv128.load offset=\d+',r'\1v128.const i32x4 0 0 0 0',text);assert n==tile//4*6,(tile,n)
 assert text.count('(')==text.count(')');return text

def main():
 p=ROOT/'scripts/build_s1_k2_latest_control_probe.py';s=p.read_text().replace('s1-k2-latest-control-v1','s1-k2-direct-v1');d=ROOT/'artifacts/s1-k2-direct-v1';d.mkdir(exist_ok=False)
 # The original wrapper uses exist_ok=False. Reserve this directory to pin the source.
 s=s.replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 patch='''
 p=src/'win_kernel.rs';text=p.read_text()
 for tile,symbol,fn in [(128,'__imajev_win7_wide_accumulate','wide'),(168,'__imajev_win7_168_accumulate','wide168'),(32,'__imajev_win7_32_accumulate','wide32')]:
  lo=text.index('#[export_name="'+symbol+'"]');hi=text.index('\\n}',lo)+2;stub=text[lo:hi]
  text+='\\n'+stub.replace(symbol,symbol+'_seed').replace('fn '+fn+'(','fn '+fn+'_seed(')
 p.write_text(text)
 p=src/'winograd.rs';text=p.read_text()
 lo=text.index(' unsafe fn simd_wide(');hi=text.index('\\n }',lo)+3
 body=text[lo:hi]
 body=body.replace('let mut out=vec![0.;q.rows()*rows];','let mut out:Vec<core::mem::MaybeUninit<f32>>=Vec::with_capacity(q.rows()*rows);out.set_len(q.rows()*rows);')
 body=body.replace('let mut sums=vec![0f32;q.rows()*tile];','')
 before='let f=match tile{168=>crate::win_kernel::wide168,32=>crate::win_kernel::wide32,_=>crate::win_kernel::wide};'
 after='let f=match (tile,b==0){(168,true)=>crate::win_kernel::wide168_seed,(168,false)=>crate::win_kernel::wide168,(32,true)=>crate::win_kernel::wide32_seed,(32,false)=>crate::win_kernel::wide32,(_,true)=>crate::win_kernel::wide_seed,_=>crate::win_kernel::wide};'
 assert body.count(before)==1;body=body.replace(before,after)
 body=body.replace('cols/256,sw.as_ptr().add(r),sums.as_mut_ptr()','rows,sw.as_ptr().add(r),out.as_mut_ptr().cast::<f32>().add(r)')
 before='for t in 0..q.rows(){out[t*rows+r..t*rows+r+tile].copy_from_slice(&sums[t*tile..(t+1)*tile]);}'
 assert body.count(before)==1;body=body.replace(before,'')
 body=body.replace('  }out','  }let mut out=core::mem::ManuallyDrop::new(out);Vec::from_raw_parts(out.as_mut_ptr().cast::<f32>(),out.len(),out.capacity())')
 text=text[:lo]+body+text[hi:];p.write_text(text)
 generated=[]
 for tile in (128,168,32):
  oldwat=old/'build'/f'kernel{tile}.wat';symbol='__imajev_win7_wide_accumulate'if tile==128 else f'__imajev_win7_{tile}_accumulate'
  for seed in (False,True):
   target=b/(f'direct{tile}'+('_seed'if seed else '')+'.wat')
   text=__import__('build_s1_k2_direct_probe').direct(oldwat.read_text(),tile,seed)
   if seed:text=text.replace(symbol,symbol+'_seed')
   target.write_text(text);generated.append((target,symbol+('_seed'if seed else '')))
'''
 anchor=" cmd=nb['command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+"\n p=src/'lib.rs'\n"+anchor)
 before="[(old/'build'/f'kernel{tile}.wat','__imajev_win7_wide_accumulate'if tile==128 else f'__imajev_win7_{tile}_accumulate')for tile in (128,168,32)]";assert s.count(before)==1;s=s.replace(before,'generated')
 (d/'frozen-builder.py').write_text(s);(d/'entry-hashes.json').write_text(json.dumps({str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()for x in [Path(__file__),p,d/'frozen-builder.py']},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
