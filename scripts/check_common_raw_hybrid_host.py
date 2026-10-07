#!/usr/bin/env python3
"""Compile the actual hybrid source with a minimal native shape/memory fixture."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=ROOT/'artifacts/update-common-raw-hybrid-v1/build';r=json.loads((b/'report.json').read_text())
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in r[key].items())
 d=b.parent/'host-check';d.mkdir(exist_ok=False)
 raw=b/'runtime/rank49_raw.rs';copy=d/'rank49_raw.rs';copy.write_text(raw.read_text()+'''\n#[cfg(test)]mod host_tests{
 use super::*;
 #[test]fn cache_and_partial_range(){
  let q=QuantizedRows::new(57,2560);let first=q.rank49_operands();assert!(core::ptr::eq(first,q.rank49_operands()));
  let len=49*first.groups*(q.cols()/4);let mut partial=vec![23456i16;len];unsafe{fill(&q,first.groups,partial.as_mut_ptr(),3,5);}
  for m in 0..49{for group in 0..first.groups{for block in 0..q.cols()/256{for k in 0..64{
   let index=(m*first.groups+group)*(q.cols()/4)+block*64+k;
   let expected:i16=A[m].iter().map(|&(i,c)|c*q.values()[(group*4+i/4)*q.cols()+block*256+(i%4)*64+k]).sum();
   assert_eq!(first.data[index],expected);assert_eq!(partial[index],if (3..5).contains(&block){expected}else{23456});
  }}}}
 }
 #[test]fn range_carry_and_raw_view_offset(){
  let q=QuantizedRows::new(57,512);let raw:Vec<u8>=(0..64*512).map(|i|((i*37+19)%256)as u8).collect();
  let fixed=crate::output_pairs::Fixed{data:super::super::pack(&raw,64,512),rows:64,cols:512};let w=PackedView{fixed,start:8*512,count:32};let sw=vec![0.015625f32;32];
  let a=q.rank49_operands();let initial:Vec<f32>=(0..57*32).map(|i|if i%3==0{-0.0}else{(i as f32-177.)/32.}).collect();
  let first=blocks(&q,&w,&sw,a.data.as_ptr(),a.groups,0,1,Some(&initial));let actual=continued(&q,&w,&sw,1,2,&first);let mut expected=initial.clone();
  for t in 0..57{for row in 0..32{for block in 0..2{let dot:i32=(0..256).map(|k|q.values()[t*512+block*256+k]as i32*(raw[(8+row)*512+block*256+k]as i8 as i32)).sum();expected[t*32+row]+=(dot as f32*q.scales()[t*2+block])*sw[row];}}}
  assert!(actual.iter().zip(&expected).all(|(a,b)|a.to_bits()==b.to_bits()));
  let seeded=blocks(&q,&w,&sw,a.data.as_ptr(),a.groups,0,2,None);let mut zero=vec![0.0f32;57*32];
  for t in 0..57{for row in 0..32{for block in 0..2{let dot:i32=(0..256).map(|k|q.values()[t*512+block*256+k]as i32*(raw[(8+row)*512+block*256+k]as i8 as i32)).sum();zero[t*32+row]+=(dot as f32*q.scales()[t*2+block])*sw[row];}}}
  assert!(seeded.iter().zip(&zero).all(|(a,b)|a.to_bits()==b.to_bits()));
 }
 #[test]fn measured_shape_selector(){
  for n in [1,8,48,56,57,87]{for(cols,rows)in[(2560,8192),(2560,9216),(4096,2560),(9216,2560),(2560,4096),(2560,512)]{
   let q=QuantizedRows::new(n,cols);let w=PackedView{fixed:crate::output_pairs::Fixed{data:vec![],rows,cols},start:0,count:rows};assert_eq!(selected(&q,&w),matches!(n,56|57)&&matches!((rows,cols),(8192,2560)|(9216,2560)|(2560,4096)|(2560,9216)));
  }}
 }
}\n''')
 # The parent source is copied solely to point its rank49 module at the appended tests.
 parent=d/'strassen_raw.rs';parent.write_text((b/'runtime/strassen_raw.rs').read_text().replace('#[path="rank49_raw.rs"]',f'#[path={json.dumps(str(copy))}]',1))
 fixture=d/'fixture.rs';fixture.write_text('''#![allow(dead_code)]
pub type Result<T>=std::result::Result<T,String>;
pub fn all_finite(v:&[f32])->bool{v.iter().all(|v|v.is_finite())}
pub mod profile{pub fn measure<T>(_:&str,f:impl FnOnce()->T)->T{f()}}
pub mod output_pairs{
 pub struct Fixed{pub data:Vec<u8>,pub rows:usize,pub cols:usize}
 pub struct PackedView{pub fixed:Fixed,pub start:usize,pub count:usize}
 impl PackedView{pub fn rows(&self)->usize{self.count}}
'''+f'#[path={json.dumps(str(parent))}]pub(crate)mod strassen_raw;\n'+'}\n'+'''
pub mod int8_kernel{
 use std::cell::OnceCell;use crate::output_pairs::strassen_raw;
 pub struct QuantizedRows{values:Vec<i16>,scales:Vec<f32>,rows:usize,cols:usize,strassen:OnceCell<strassen_raw::Operands>,rank49:OnceCell<strassen_raw::rank49::Operands>}
 impl QuantizedRows{
  pub fn new(rows:usize,cols:usize)->Self{let padded=rows.div_ceil(8)*8;let mut values:Vec<i16>=(0..padded*cols).map(|i|((i*73+17)%255)as i16-127).collect();values[rows*cols..].fill(0);Self{values,scales:(0..padded*(cols/256)).map(|i|0.0078125*(1+(i%3))as f32).collect(),rows,cols,strassen:OnceCell::new(),rank49:OnceCell::new()}}
  pub fn rows(&self)->usize{self.rows}pub fn cols(&self)->usize{self.cols}pub fn values(&self)->&[i16]{&self.values}pub fn scales(&self)->&[f32]{&self.scales}
  pub(crate)fn strassen_operands(&self)->&strassen_raw::Operands{self.strassen.get_or_init(||strassen_raw::Operands::new(self))}
  pub(crate)fn rank49_operands(&self)->&strassen_raw::rank49::Operands{self.rank49.get_or_init(||strassen_raw::rank49::Operands::new(self))}
 }
}
''')
 command=['rustc','--edition=2021','--test',str(fixture),'-C','opt-level=3','--cfg','feature="experimental-int8-k-continue"','-o',str(d/'check')]
 result=subprocess.run(command,capture_output=True,text=True);(d/'compiler.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 result=subprocess.run([str(d/'check'),'--test-threads=1'],capture_output=True,text=True);(d/'test.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 files=[Path(__file__),b/'report.json',raw,b/'runtime/strassen_raw.rs',copy,parent,fixture,d/'compiler.log',d/'test.log',d/'check'];report=dict(complete=True,tests=3,actual_generated_sources_compiled=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Native fixture only: exact actual rank49 input cache/range-fill/selector code; raw-view offset and ascendingK scalar backend bit oracle with carry/seed. Minimal shape/accessor fixture substitutes production QuantizedRows/PackedView; not SIMD, graph or paid fidelity proof.')
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(result.stdout)
if __name__=='__main__':main()
