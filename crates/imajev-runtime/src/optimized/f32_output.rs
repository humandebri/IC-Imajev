//! Query-local output-lane SIMD with original F32 column accumulation order.
use crate::{prepared_weights::PreparedF32, Result};
pub(crate) fn project(
    x: &[f32],
    w: &PreparedF32,
    n: usize,
    rows: usize,
    cols: usize,
) -> Option<Result<Vec<f32>>> {
    let (packed, start, prepared_cols) = w.output_view()?;
    if n > 132 {
        return None;
    }
    Some((|| {
        if cols != prepared_cols
            || n == 0
            || rows == 0
            || n.checked_mul(cols) != Some(x.len())
            || rows.checked_mul(cols) != Some(w.len())
            || n.checked_mul(rows).is_none_or(|v| v > crate::MAX_FLOATS)
            || !crate::all_finite(&x)
        {
            return Err("output F32 projection shape/finite".into());
        }
        #[cfg(target_arch="wasm32")]
        if rows%64==0 && start%32==0 && cols%64==0 {return project_wide(x,packed,start,n,rows,cols); }
        let mut out = vec![0.; n * rows];
        #[cfg(target_arch = "wasm32")]
        for r in (0..rows).step_by(32) {
            let width = (rows - r).min(32);
            let global = start + r;
            let pad;
            let weight = if width == 32 && global % 32 == 0 {
                &packed[global * cols..(global + 32) * cols]
            } else {
                pad = {
                    let mut values = vec![0.; 32 * cols];
                    for c in 0..cols {
                        for j in 0..width {
                            let row = global + j;
                            values[c * 32 + j] = packed[(row / 32) * 32 * cols + c * 32 + row % 32];
                        }
                    }
                    values
                };
                &pad
            };
            let mut sums = vec![[0f32; 32]; n];
            // Both private prepared constructors establish the complete fixed-weight
            // tile, and query validation establishes every real input token. The
            // patched body writes all n*32 initialized F32 output cells.
            for column in (0..cols).step_by(64) {
                unsafe {
                    accumulate(
                        x.as_ptr(),
                        weight.as_ptr(),
                        cols,
                        column,
                        x.as_ptr(),
                        32,
                        weight.as_ptr(),
                        sums.as_mut_ptr().cast(),
                        n,
                    );
                }
            }
            for t in 0..n {
                out[t * rows + r..t * rows + r + width].copy_from_slice(&sums[t][..width]);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        for t in 0..n {
            for r in 0..rows {
                let global = start + r;
                for c in 0..cols {
                    out[t * rows + r] +=
                        x[t * cols + c] * packed[(global / 32) * 32 * cols + c * 32 + global % 32];
                }
            }
        }
        if !crate::all_finite(&out) {
            return Err("output F32 projection finite".into());
        }
        Ok(out)
    })())
}
#[cfg(target_arch = "wasm32")]
#[export_name = "__imajev_f32_output64"]
#[inline(never)]
unsafe extern "C" fn accumulate(
    q: *const f32,
    w: *const f32,
    cols: usize,
    start: usize,
    sx: *const f32,
    stride: usize,
    sw: *const f32,
    out: *mut f32,
    n: usize,
) {
    let marker = core::hint::black_box(
        (q as usize)
            ^ (w as usize)
            ^ cols
            ^ start
            ^ (sx as usize)
            ^ stride
            ^ (sw as usize)
            ^ (out as usize)
            ^ n,
    ) as u32;
    for i in 0..n * 32 {
        core::ptr::write_volatile(out.add(i), core::hint::black_box(f32::from_bits((marker ^ 3) | 0x7fc00000)));
    }
}

/// Resume an original-order product from complete64-column cuts. The input
/// chunk is token-major; fixed weights retain their original packed stride.
#[cfg(feature="experimental-f32-k-continue")]
pub(crate) fn continue_columns(x:&[f32],initial:&[f32],w:&PreparedF32,n:usize,rows:usize,cols:usize,begin:usize,count:usize)->Result<Vec<f32>> {
 let(packed,start,prepared_cols)=w.output_view().ok_or("F32 continuation layout")?;
 if n==0 || n>132 || rows==0 || cols!=prepared_cols || rows.checked_mul(cols)!=Some(w.len())
  || n.checked_mul(count)!=Some(x.len()) || n.checked_mul(rows)!=Some(initial.len()) || initial.len()>crate::MAX_FLOATS
  || count==0 || count%64!=0 || begin%64!=0 || begin.checked_add(count).is_none_or(|v|v>cols)
  || !(crate::all_finite(x) && crate::all_finite(initial)) {return Err("F32 continuation span/finite".into());}
 let mut out=initial.to_vec();
 #[cfg(target_arch="wasm32")]
 for r in(0..rows).step_by(32) {
  let width=(rows-r).min(32);let global=start+r;let pad;
  let weight=if width==32 && global%32==0 {&packed[global*cols..(global+32)*cols]}else{
   pad={let mut p=vec![0.;32*cols];for c in 0..cols{for j in 0..width{let row=global+j;p[c*32+j]=packed[(row/32)*32*cols+c*32+row%32];}}p};&pad
  };
  let mut sums=vec![[0f32;32];n];for t in 0..n{sums[t][..width].copy_from_slice(&initial[t*rows+r..t*rows+r+width]);}
  // All pointers remain within their initialized allocations. Shifting the
  // weight base by begin*32 makes local column0 mean original column begin;
  // using count as input stride addresses the compact token-major chunk.
  let weight_base=unsafe{weight.as_ptr().add(begin*32)};
  for column in(0..count).step_by(64){unsafe{accumulate(x.as_ptr(),weight_base,count,column,x.as_ptr(),32,weight_base,sums.as_mut_ptr().cast(),n);}}
  for t in 0..n{out[t*rows+r..t*rows+r+width].copy_from_slice(&sums[t][..width]);}
 }
 #[cfg(not(target_arch="wasm32"))]
 for t in 0..n{for r in 0..rows{let global=start+r;for c in 0..count{out[t*rows+r]+=x[t*count+c]*packed[(global/32)*32*cols+(begin+c)*32+global%32];}}}
 if !crate::all_finite(&out){return Err("F32 continuation result finite".into());}Ok(out)
}
#[cfg(all(test,feature="experimental-f32-k-continue"))]
mod continuation_tests {
 use super::*;
 #[test]fn every_cut_preserves_full_scalar_order_with_extremes_cancellation_and_partial_output_rows(){
  for cols in [128,2560,9216]{let rows=64;let raw:Vec<f32>=(0..rows*cols).map(|i|[1.,-1.,-0.,0.,0.1234567,f32::MIN_POSITIVE][i%6]).collect();let bytes:Vec<_>=raw.iter().flat_map(|x|x.to_le_bytes()).collect();let fixed=PreparedF32::from_le_bytes_output(&bytes,rows,cols).unwrap();
   for(n,start,width)in[(1,0,64),(7,2,8),(45,31,33),(87,32,32)]{let view=fixed.slice(start*cols,width*cols).unwrap();let x:Vec<f32>=(0..n*cols).map(|i|[1.,-1.,-0.,0.,1e-30,0.2345678][i%6]).collect();let expected=crate::matrix_reference(&x,&raw[start*cols..(start+width)*cols],n,width,cols).unwrap();
    for chunk in [64,128,256,1024]{let mut out=vec![0.;n*width];for begin in(0..cols).step_by(chunk){let count=(cols-begin).min(chunk);let mut part=Vec::with_capacity(n*count);for t in 0..n{part.extend_from_slice(&x[t*cols+begin..t*cols+begin+count]);}out=continue_columns(&part,&out,&view,n,width,cols,begin,count).unwrap();}assert!(out.iter().zip(&expected).all(|(a,b)|a.to_bits()==b.to_bits()),"cols={cols} n={n} chunk={chunk}");}
   }
  }
 }
}

// Original packed output32 stride; group two adjacent tiles while preserving
// ascending F32 multiply/add order in every scalar output cell.
#[cfg(target_arch="wasm32")]
fn project_wide(x:&[f32],packed:&[f32],start:usize,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {
 if rows%512==0 && cols==64 {return project_wide512(x,packed,start,n,rows,cols);}
 if rows==64 && cols%512==0 {return project_columns512(x,packed,start,n,rows,cols);}
 if rows%128==0 {return project_wide128(x,packed,start,n,rows,cols);}
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
 if !crate::all_finite(&out){return Err("F32 output64 finite".into());}Ok(out)
}
#[cfg(target_arch="wasm32")]
#[export_name="__imajev_f32_wide"]
#[inline(never)]
unsafe extern "C" fn accumulate_wide(q:*const f32,w:*const f32,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize) {
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;
 for i in 0..n*64 {core::ptr::write_volatile(out.add(i),core::hint::black_box(f32::from_bits((marker^5)|0x7fc00000)));}
}

// Original packed output32 stride; group two adjacent tiles while preserving
// ascending F32 multiply/add order in every scalar output cell.
#[cfg(target_arch="wasm32")]
fn project_wide128(x:&[f32],packed:&[f32],start:usize,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {
 if rows%128!=0 || start%32!=0 || cols%64!=0 {return Err("F32 output128 alignment".into());}
 let mut out=vec![0.;n*rows];
 for r in(0..rows).step_by(128) {
  let global=start+r;
  let weight=&packed[global*cols..(global+128)*cols];
  let mut sums=vec![[0f32;128];n];
  for column in(0..cols).step_by(64) {
   unsafe{accumulate_wide128(x.as_ptr(),weight.as_ptr(),cols,column,x.as_ptr(),128,weight.as_ptr(),sums.as_mut_ptr().cast(),n);}
  }
  for t in 0..n{out[t*rows+r..t*rows+r+128].copy_from_slice(&sums[t]);}
 }
 if !crate::all_finite(&out){return Err("F32 output128 finite".into());}Ok(out)
}
#[cfg(target_arch="wasm32")]
#[export_name="__imajev_f32_wide128"]
#[inline(never)]
unsafe extern "C" fn accumulate_wide128(q:*const f32,w:*const f32,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize) {
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;
 for i in 0..n*128 {core::ptr::write_volatile(out.add(i),core::hint::black_box(f32::from_bits((marker^517)|0x7fc00000)));}
}

#[cfg(target_arch="wasm32")]
fn project_wide512(x:&[f32],packed:&[f32],start:usize,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {
 if rows%512!=0 || start%32!=0 || cols%64!=0 {return Err("F32 output512 alignment".into());}
 let mut out=vec![0.;n*rows];
 for r in(0..rows).step_by(512) {
  let global=start+r;
  let weight=&packed[global*cols..(global+512)*cols];
  let mut sums=vec![[0f32;512];n];
  for column in(0..cols).step_by(64) {
   unsafe{accumulate_wide512(x.as_ptr(),weight.as_ptr(),cols,column,x.as_ptr(),512,weight.as_ptr(),sums.as_mut_ptr().cast(),n);}
  }
  for t in 0..n{out[t*rows+r..t*rows+r+512].copy_from_slice(&sums[t]);}
 }
 if !crate::all_finite(&out){return Err("F32 output512 finite".into());}Ok(out)
}
#[cfg(target_arch="wasm32")]
#[export_name="__imajev_f32_wide512"]
#[inline(never)]
unsafe extern "C" fn accumulate_wide512(q:*const f32,w:*const f32,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize) {
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;
 for i in 0..n*512 {core::ptr::write_volatile(out.add(i),core::hint::black_box(f32::from_bits((marker^1517)|0x7fc00000)));}
}

#[cfg(target_arch="wasm32")]
fn project_columns512(x:&[f32],packed:&[f32],start:usize,n:usize,rows:usize,cols:usize)->Result<Vec<f32>> {

 if rows%64!=0 || start%32!=0 || cols%64!=0 {return Err("F32 output64 alignment".into());}
 let mut out=vec![0.;n*rows];
 for r in(0..rows).step_by(64) {
  let global=start+r;
  let weight=&packed[global*cols..(global+64)*cols];
  let mut sums=vec![[0f32;64];n];
  for column in(0..cols).step_by(512) {
   unsafe{accumulate_columns512(x.as_ptr(),weight.as_ptr(),cols,column,x.as_ptr(),64,weight.as_ptr(),sums.as_mut_ptr().cast(),n);}
  }
  for t in 0..n{out[t*rows+r..t*rows+r+64].copy_from_slice(&sums[t]);}
 }
 if !crate::all_finite(&out){return Err("F32 output64 finite".into());}Ok(out)
}
#[cfg(target_arch="wasm32")]
#[export_name="__imajev_f32_columns512"]
#[inline(never)]
unsafe extern "C" fn accumulate_columns512(q:*const f32,w:*const f32,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,out:*mut f32,n:usize) {
 let marker=core::hint::black_box((q as usize)^(w as usize)^cols^start^(sx as usize)^stride^(sw as usize)^(out as usize)^n)as u32;
 for i in 0..n*64 {core::ptr::write_volatile(out.add(i),core::hint::black_box(f32::from_bits((marker^1518)|0x7fc00000)));}
}

// Original packed output32 stride; group two adjacent tiles while preserving
// ascending F32 multiply/add order in every scalar output cell.
