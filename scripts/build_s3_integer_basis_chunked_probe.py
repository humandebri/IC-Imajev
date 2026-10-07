#!/usr/bin/env python3
"""Bound immutable-only coefficient preparation to 512 rows per update."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'artifacts/s3-integer-basis-probe-entry-v1/frozen-builder.py';d=ROOT/'artifacts/s3-integer-basis-chunked-probe-v1';d.mkdir(exist_ok=False)
 s=old.read_text().replace('s3-integer-basis-probe-v1','s3-integer-basis-chunked-probe-v1').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 s=s.replace('pub struct Prepared{data:Vec<i16>,rows:usize,cols:usize}','pub struct Prepared{data:Vec<i16>,rows:usize,cols:usize,pub next:usize}')
 before='''  let blocks=cols/256;let mut data=vec![0i16;(rows/32)*blocks*343*128];
  for group in 0..rows/32{for block in 0..blocks{for m in 0..343{for k in 0..32{for lane in 0..4{
'''
 after='''  let blocks=cols/256;let data=vec![0i16;(rows/32)*blocks*343*128];
  Ok(Self{data,rows,cols,next:0})
 }
 pub fn prepare_chunk(&mut self,w:&[i8],count:usize){
  let rows=self.rows;let cols=self.cols;let blocks=cols/256;assert_eq!(w.len(),rows*cols);assert!(count>0&&count%32==0&&self.next+count<=rows);
  let data=&mut self.data;
  for group in self.next/32..(self.next+count)/32{for block in 0..blocks{for m in 0..343{for k in 0..32{for lane in 0..4{
'''
 assert s.count(before)==1;s=s.replace(before,after)
 assert s.count('  Ok(Self{data,rows,cols})')==1;s=s.replace('  Ok(Self{data,rows,cols})','  self.next+=count;')
 before=" (src/'lib.rs').write_text(lib)"
 injection=''' lib=lib.replace('f.win=Some(s2::Prepared::new(&f.w,f.rows,f.cols).unwrap());','assert_eq!(f.win.as_ref().unwrap().next,f.rows);')
 lib+='\\n#[ic_cdk::update]fn prepare_coefficients()->Preparation{FIXED.with(|s|{let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(!f.sealed&&f.scales.len()==f.rows);let begin=ic_cdk::api::performance_counter(0);if f.win.is_none(){f.win=Some(s2::Prepared::new(&f.w,f.rows,f.cols).unwrap());}let next=f.win.as_ref().unwrap().next;let count=(f.rows-next).min(512);f.win.as_mut().unwrap().prepare_chunk(&f.w,count);Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:f.win.as_ref().unwrap().bytes()as u64,rows:count as u64}})}\\n'
'''
 assert s.count(before)==1;s=s.replace(before,injection+before)
 frozen=d/'frozen-builder.py';frozen.write_text(s)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),old,frozen]},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
