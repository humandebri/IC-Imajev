#!/usr/bin/env python3
"""Compile actual new source with tiny type adapters; independently check layout and carry."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/update-k2-compact-v1/native-check';d.mkdir(exist_ok=False)
 s=(ROOT/'scripts/strassen_k2_runtime.rs').read_text()
 s+='''
 pub(crate) fn audit_operands(q:&QuantizedRows){
  let a=q.strassen_operands();let mut seen=vec![0u8;a.data.len()];let cols=q.cols();
  for m in 0..7{for pair in 0..a.pairs{for b in 0..cols/256{for k in 0..128{
   let p=pair*2*cols+b*256+k;let(v0,v1,v2,v3)=(q.values()[p],q.values()[p+128],q.values()[p+cols],q.values()[p+cols+128]);
   let expected=[v0,v1,v0+v1-v2-v3,v3,v2+v3,-v0+v2+v3,v0-v2][m];
   let i=m*a.pairs*(cols/2)+pair*(cols/2)+b*128+k;assert_eq!(a.data[i],expected);seen[i]+=1;
  }}}}assert!(seen.iter().all(|v|*v==1));
 }
'''
 (d/'strassen_raw.rs').write_text(s)
 harness='''
 type Result<T>=std::result::Result<T,String>;
 fn all_finite(v:&[f32])->bool{v.iter().all(|x|x.is_finite())}
 mod profile{pub fn measure<T>(_:&str,f:impl FnOnce()->T)->T{f()}}
 mod int8_kernel{
  pub struct QuantizedRows{pub v:Vec<i16>,pub s:Vec<f32>,pub rows:usize,pub cols:usize,pub a:std::cell::OnceCell<crate::output_pairs::strassen_raw::Operands>}
  impl QuantizedRows{pub fn rows(&self)->usize{self.rows}pub fn cols(&self)->usize{self.cols}pub fn values(&self)->&[i16]{&self.v}pub fn scales(&self)->&[f32]{&self.s}pub fn strassen_operands(&self)->&crate::output_pairs::strassen_raw::Operands{self.a.get_or_init(||crate::output_pairs::strassen_raw::Operands::new(self))}}
 }
 mod output_pairs{
  pub struct PreparedPairs{pub data:Vec<u8>,pub rows:usize}
  pub struct PackedView{pub fixed:PreparedPairs,pub start:usize,pub rows:usize,pub cols:usize}
  impl PackedView{pub fn rows(&self)->usize{self.rows}}
  #[path="STRASSEN"]pub mod strassen_raw;
 }
 fn reference(q:&int8_kernel::QuantizedRows,original:&[u8],start:usize,rows:usize,sw:&[f32],first:usize,end:usize,mut out:Vec<f32>)->Vec<f32>{
  for t in 0..q.rows{for r in 0..rows{for b in first..end{let mut dot=0i32;for k in 0..256{dot+=q.v[t*q.cols+b*256+k]as i32*(original[(start+r)*q.cols+b*256+k]as i8 as i32);}out[t*rows+r]=out[t*rows+r]+((dot as f32)*q.s[t*(q.cols/256)+b])*sw[r];}}}out
 }
 fn bits(v:&[f32])->Vec<u32>{v.iter().map(|x|x.to_bits()).collect()}
 fn main(){let mut conditions=0;let mut decoded=0;
  for cols in [256,512,2560,9216]{let total=512;let original:Vec<u8>=(0..total*cols).map(|i|((i*73+i/11)%256)as u8).collect();let mut raw=original.clone();raw.extend(vec![0;total*4]);
   let packed=output_pairs::strassen_raw::pack(&raw,total,cols);assert_eq!(&packed[total*cols..],&raw[total*cols..]);
   let mut seen=vec![false;total*cols];for r in 0..total{for c in 0..cols{let p=output_pairs::strassen_raw::index(total,cols,r,c);assert!(!seen[p]);seen[p]=true;assert_eq!(packed[p],original[r*cols+c]);decoded+=1;}}assert!(seen.iter().all(|x|*x));
   for n in [1,2,3,5,8,9]{let mut v=vec![0i16;n.div_ceil(8)*8*cols];for i in 0..n*cols{v[i]=[-127,127,0,-1,1,61,-61][i%7];}
    let q=int8_kernel::QuantizedRows{v,s:(0..n*(cols/256)).map(|i|[0.001f32,1.,0.125,1.234567][i%4]).collect(),rows:n,cols,a:std::cell::OnceCell::new()};output_pairs::strassen_raw::audit_operands(&q);
    for (start,rows) in [(0,512),(0,128),(2,8),(6,32),(8,128)]{
     let sw:Vec<f32>=(0..rows).map(|i|[0.001f32,0.1234567,1.,0.125][i%4]).collect();let w=output_pairs::PackedView{fixed:output_pairs::PreparedPairs{data:packed[..total*cols].to_vec(),rows:total},start:start*cols,rows,cols};
     let expected=reference(&q,&original,start,rows,&sw,0,cols/256,vec![0.;n*rows]);let actual=output_pairs::strassen_raw::project(&q,&w,&sw).unwrap();assert_eq!(bits(&expected),bits(&actual));conditions+=1;
     let cut=if cols>=2560{5}else{1};if cut<cols/256{
      let initial:Vec<f32>=(0..n*rows).map(|i|[0.,-0.,0.1234567,-13.75][i%4]).collect();let expected=reference(&q,&original,start,rows,&sw,cut,cols/256,initial.clone());
      let actual=output_pairs::strassen_raw::project_continued(&q,&w,&sw,cut*256,cols-cut*256,&initial).unwrap();assert_eq!(bits(&expected),bits(&actual));conditions+=1;
      let carry=reference(&q,&original,start,rows,&sw,0,cut,vec![0.;n*rows]);let joined=output_pairs::strassen_raw::project_continued(&q,&w,&sw,cut*256,cols-cut*256,&carry).unwrap();assert_eq!(bits(&joined),bits(&actual_full(&q,&w,&sw)));conditions+=1;
     }
    }
   }
  }println!("{{\\"conditions\\":{},\\"decoded_weight_bytes\\":{},\\"native_bits_equal\\":true}}",conditions,decoded);
 }
 fn actual_full(q:&int8_kernel::QuantizedRows,w:&output_pairs::PackedView,sw:&[f32])->Vec<f32>{output_pairs::strassen_raw::project(q,w,sw).unwrap()}
'''.replace('STRASSEN',str(d/'strassen_raw.rs')).replace('for n in [1,2,3,5,8,9]','for n in [1usize,2,3,5,8,9]')
 (d/'main.rs').write_text(harness)
 command=['rustc','--edition=2021','-O','--cfg','feature="experimental-int8-k-continue"',str(d/'main.rs'),'-o',str(d/'check')]
 with (d/'compiler.log').open('w')as f:subprocess.run(command,stdout=f,stderr=f,check=True)
 result=json.loads(subprocess.check_output([str(d/'check')],text=True));result.update(command=command,scope=__doc__,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'scripts/strassen_k2_runtime.rs',d/'strassen_raw.rs',d/'main.rs',d/'check']})
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
