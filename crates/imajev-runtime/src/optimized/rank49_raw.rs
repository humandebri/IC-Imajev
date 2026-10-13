use crate::int8_kernel::QuantizedRows;
use crate::output_pairs::PackedView;
#[cfg(target_arch="wasm32")]#[path="rank49_prepare.rs"]mod prepare;
#[cfg(target_arch="wasm32")]#[path="rank49_kernel.rs"]mod kernel;
#[derive(Debug)]pub(crate)struct Operands{data:Vec<i16>,groups:usize}
impl Operands{
 pub(crate)fn new(q:&QuantizedRows)->Self{
  let groups=q.rows().div_ceil(4);let len=49*groups*(q.cols()/4);let mut data=Vec::<i16>::with_capacity(len);
  unsafe{fill(q,groups,data.as_mut_ptr(),0,q.cols()/256);data.set_len(len);}Self{data,groups}
 }
}
pub(super)fn selected(q:&QuantizedRows,w:&PackedView)->bool{
 matches!(q.rows(),56|57)&&w.start/q.cols()%8==0&&matches!((w.rows(),q.cols()),(8192,2560)|(9216,2560)|(2560,4096)|(2560,9216))
}
unsafe fn fill(q:&QuantizedRows,groups:usize,out:*mut i16,first:usize,end:usize){
 #[cfg(target_arch="wasm32")]prepare::inputs(q,groups,out,first,end);
 #[cfg(not(target_arch="wasm32"))]for group in 0..groups{for block in first..end{for k in 0..64{for m in 0..49{
  let value:i16=A[m].iter().map(|&(i,c)|c*q.values()[(group*4+i/4)*q.cols()+block*256+(i%4)*64+k]).sum();
  out.add((m*groups+group)*(q.cols()/4)+block*64+k).write(value);
 }}}}
}
pub(super)fn project(q:&QuantizedRows,w:&PackedView,sw:&[f32])->Vec<f32>{
 let a=q.rank49_operands();blocks(q,w,sw,a.data.as_ptr(),a.groups,0,q.cols()/256,None)
}
#[cfg(feature="experimental-int8-k-continue")]
pub(super)fn continued(q:&QuantizedRows,w:&PackedView,sw:&[f32],first:usize,end:usize,initial:&[f32])->Vec<f32>{
 let groups=q.rows().div_ceil(4);let len=49*groups*(q.cols()/4);
 let data=crate::profile::measure("integer_k_prepare_new_columns",||{
  let mut data=Vec::<core::mem::MaybeUninit<i16>>::with_capacity(len);
  unsafe{data.set_len(len);fill(q,groups,data.as_mut_ptr().cast(),first,end);}data
 });
 blocks(q,w,sw,data.as_ptr().cast(),groups,first,end,Some(initial))
}
fn blocks(q:&QuantizedRows,w:&PackedView,sw:&[f32],data:*const i16,groups:usize,first:usize,end:usize,initial:Option<&[f32]>)->Vec<f32>{
 let cols=q.cols();let rows=w.rows();let start=w.start/cols;let fixed=&w.fixed;
 #[cfg(not(target_arch="wasm32"))]{
  let _=(data,groups);let mut out=initial.map_or_else(||vec![0.;q.rows()*rows],|v|v.to_vec());
  for t in 0..q.rows(){for r in 0..rows{for block in first..end{
   let dot:i32=(0..256).map(|k|q.values()[t*cols+block*256+k]as i32*(fixed.data[super::index(fixed.rows,cols,start+r,block*256+k)]as i8 as i32)).sum();
   out[t*rows+r]+=(dot as f32*q.scales()[t*(cols/256)+block])*sw[r];
  }}}out
 }
 #[cfg(target_arch="wasm32")]unsafe{
  let len=q.rows()*rows;let mut out=Vec::<core::mem::MaybeUninit<f32>>::with_capacity(len);out.set_len(len);let output=out.as_mut_ptr().cast::<f32>();
  if let Some(v)=initial{core::ptr::copy_nonoverlapping(v.as_ptr(),output,len);}
  let ap:[*const i16;49]=core::array::from_fn(|m|data.add(m*groups*(cols/4)));let plane=fixed.rows*cols/4;let mut r=0;
  while r<rows{let tile=if rows-r>=96{96}else{16};
   let wp:[*const i8;16]=core::array::from_fn(|m|fixed.data.as_ptr().add(((m/4/2)*2+(m%4)%2)*plane+((start+r)/8+(m%4)/2)*cols*2+((m/4)%2)*256).cast());
   for block in first..end{let f=match(tile,initial.is_none()&&block==first){(96,true)=>kernel::tile96_seed,(96,false)=>kernel::tile96,(16,true)=>kernel::tile16_seed,_=>kernel::tile16};
    f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),rows,sw.as_ptr().add(r),output.add(r),q.rows());
   }r+=tile;
  }
  let mut out=core::mem::ManuallyDrop::new(out);Vec::from_raw_parts(out.as_mut_ptr().cast(),out.len(),out.capacity())
 }
}
const A:[&[(usize,i16)];49]=[
&[(0,1)],
&[(1,1)],
&[(0,1),(1,1),(4,-1),(5,-1)],
&[(5,1)],
&[(4,1),(5,1)],
&[(0,-1),(4,1),(5,1)],
&[(0,1),(4,-1)],
&[(2,1)],
&[(3,1)],
&[(2,1),(3,1),(6,-1),(7,-1)],
&[(7,1)],
&[(6,1),(7,1)],
&[(2,-1),(6,1),(7,1)],
&[(2,1),(6,-1)],
&[(0,1),(2,1),(8,-1),(10,-1)],
&[(1,1),(3,1),(9,-1),(11,-1)],
&[(0,1),(1,1),(2,1),(3,1),(4,-1),(5,-1),(6,-1),(7,-1),(8,-1),(9,-1),(10,-1),(11,-1),(12,1),(13,1),(14,1),(15,1)],
&[(5,1),(7,1),(13,-1),(15,-1)],
&[(4,1),(5,1),(6,1),(7,1),(12,-1),(13,-1),(14,-1),(15,-1)],
&[(0,-1),(2,-1),(4,1),(5,1),(6,1),(7,1),(8,1),(10,1),(12,-1),(13,-1),(14,-1),(15,-1)],
&[(0,1),(2,1),(4,-1),(6,-1),(8,-1),(10,-1),(12,1),(14,1)],
&[(10,1)],
&[(11,1)],
&[(10,1),(11,1),(14,-1),(15,-1)],
&[(15,1)],
&[(14,1),(15,1)],
&[(10,-1),(14,1),(15,1)],
&[(10,1),(14,-1)],
&[(8,1),(10,1)],
&[(9,1),(11,1)],
&[(8,1),(9,1),(10,1),(11,1),(12,-1),(13,-1),(14,-1),(15,-1)],
&[(13,1),(15,1)],
&[(12,1),(13,1),(14,1),(15,1)],
&[(8,-1),(10,-1),(12,1),(13,1),(14,1),(15,1)],
&[(8,1),(10,1),(12,-1),(14,-1)],
&[(0,-1),(8,1),(10,1)],
&[(1,-1),(9,1),(11,1)],
&[(0,-1),(1,-1),(4,1),(5,1),(8,1),(9,1),(10,1),(11,1),(12,-1),(13,-1),(14,-1),(15,-1)],
&[(5,-1),(13,1),(15,1)],
&[(4,-1),(5,-1),(12,1),(13,1),(14,1),(15,1)],
&[(0,1),(4,-1),(5,-1),(8,-1),(10,-1),(12,1),(13,1),(14,1),(15,1)],
&[(0,-1),(4,1),(8,1),(10,1),(12,-1),(14,-1)],
&[(0,1),(8,-1)],
&[(1,1),(9,-1)],
&[(0,1),(1,1),(4,-1),(5,-1),(8,-1),(9,-1),(12,1),(13,1)],
&[(5,1),(13,-1)],
&[(4,1),(5,1),(12,-1),(13,-1)],
&[(0,-1),(4,1),(5,1),(8,1),(12,-1),(13,-1)],
&[(0,1),(4,-1),(8,-1),(12,1)],
];
