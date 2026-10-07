//! Share projections only for identical quantized lanes, scale bits and both LoRA-A rows.
//! Stable first occurrence preserves ordering; outputs are expanded before contextual work.
pub(super) struct RowPlan {pub(super) first:Vec<usize>,pub(super) index:Vec<usize>}
fn bits_equal(a:&[f32],b:&[f32])->bool {a.len()==b.len()&&a.iter().zip(b).all(|(a,b)|a.to_bits()==b.to_bits())}
impl RowPlan {
 pub(super) fn new(q:&[i16],sx:&[f32],qa:&[f32],za:&[f32],n:usize,cols:usize,rank:usize)->Option<Self>{
  assert!(n>0&&cols%256==0&&q.len()>=n*cols&&sx.len()>=n*(cols/256)&&qa.len()==n*rank&&za.len()==n*rank);
  let blocks=cols/256;let mut first=Vec::new();let mut index=Vec::with_capacity(n);
  for t in 0..n {
   let found=first.iter().position(|&u|q[t*cols..(t+1)*cols]==q[u*cols..(u+1)*cols]
    &&bits_equal(&sx[t*blocks..(t+1)*blocks],&sx[u*blocks..(u+1)*blocks])
    &&bits_equal(&qa[t*rank..(t+1)*rank],&qa[u*rank..(u+1)*rank])
    &&bits_equal(&za[t*rank..(t+1)*rank],&za[u*rank..(u+1)*rank]));
   index.push(match found{Some(i)=>i,None=>{first.push(t);first.len()-1}});
  }
  if first.len()==n {None}else{Some(Self{first,index})}
 }
 pub(super) fn gather(&self,x:&[f32],cols:usize)->Vec<f32>{
  assert_eq!(x.len(),self.index.len()*cols);let mut out=Vec::with_capacity(self.first.len()*cols);
  for &t in &self.first{out.extend_from_slice(&x[t*cols..(t+1)*cols]);}out
 }
 pub(super) fn expand(&self,x:&[f32],cols:usize)->Vec<f32>{
  assert_eq!(x.len(),self.first.len()*cols);let mut out=Vec::with_capacity(self.index.len()*cols);
  for &t in &self.index{out.extend_from_slice(&x[t*cols..(t+1)*cols]);}out
 }
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn stable_exact_mapping_and_roundtrip(){
  let q=vec![3;5*256];let sx=vec![1.;5];let qa=vec![2.,4.,2.,5.,4.];let za=vec![8.;5];
  let p=RowPlan::new(&q,&sx,&qa,&za,5,256,1).unwrap();assert_eq!(p.first,[0,1,3]);assert_eq!(p.index,[0,1,0,2,1]);
  let x=vec![2.,-0.,4.,0.,2.,-0.,5.,7.,4.,0.];let restored=p.expand(&p.gather(&x,2),2);
  assert!(bits_equal(&restored,&x));
 }
 #[test]fn each_input_plane_prevents_false_sharing(){
  let mut q=vec![0;2*256];let mut sx=vec![1.;2];let mut qa=vec![0.;2];let mut za=vec![0.;2];
  assert!(RowPlan::new(&q,&sx,&qa,&za,2,256,1).is_some());q[511]=1;assert!(RowPlan::new(&q,&sx,&qa,&za,2,256,1).is_none());q[511]=0;
  sx[1]=f32::from_bits(1f32.to_bits()+1);assert!(RowPlan::new(&q,&sx,&qa,&za,2,256,1).is_none());sx[1]=1.;
  qa[1]=-0.;assert!(RowPlan::new(&q,&sx,&qa,&za,2,256,1).is_none());qa[1]=0.;
  za[1]=f32::from_bits(1);assert!(RowPlan::new(&q,&sx,&qa,&za,2,256,1).is_none());
 }
 #[test]fn every_cardinality_and_nonadjacent_repeat(){
  for n in 1..=89{for distinct in 1..=n{
   let mut q=vec![0;256*n];for t in 0..n{q[t*256]=(t%distinct)as i16;}
   let sx=vec![1.;n];let a=vec![0.;n*64];let p=RowPlan::new(&q,&sx,&a,&a,n,256,64);
   if distinct==n{assert!(p.is_none());}else{let p=p.unwrap();assert_eq!(p.first.len(),distinct);assert_eq!(p.index,(0..n).map(|t|t%distinct).collect::<Vec<_>>());}
  }}
 }
}
