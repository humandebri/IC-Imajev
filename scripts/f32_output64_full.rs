// Original packed output32 stride; group two adjacent tiles while preserving
// ascending F32 multiply/add order in every scalar output cell.
#[cfg(target_arch="wasm32")]
fn project_wide(x:&[f32],packed:&[f32],start:usize,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {
 if rows%64!=0 || start%32!=0 || cols%64!=0 {return Err("F32 output64 alignment".into());}
 let mut out=vec![0.;n*rows];
 for r in(0..rows).step_by(64) {
  let global=start+r;
  let weight=&packed[global*cols..(global+64)*cols];
  let mut sums=vec![[0f32;64];n];
  for column in(0..cols).step_by(64) {
   unsafe{accumulate_wide(x.as_ptr(),weight.as_ptr(),cols,column,x.as_ptr(),64,weight.as_ptr(),sums.as_mut_ptr().cast(),n);}
  }
  for t in 0..n{out[t*rows+r..t*rows+r+64].copy_from_slice(&sums[t]);}
 }
 if !out.iter().all(|v|v.is_finite()){return Err("F32 output64 finite".into());}Ok(out)
}
#[cfg(target_arch="wasm32")]
#[export_name="__imajev_f32_wide"]
#[inline(never)]
unsafe extern "C" fn accumulate_wide(q:*const f32,w:*const f32,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize) {
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;
 for i in 0..n*64 {core::ptr::write_volatile(out.add(i),core::hint::black_box(f32::from_bits((marker^5)|0x7fc00000)));}
}
