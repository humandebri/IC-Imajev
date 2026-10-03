//! Experimental exact pair-factor projection; independent of the adopted runtime.
use imajev_runtime::int8_kernel::QuantizedRows;
pub fn prepare(w:&[i8],cols:usize)->(Vec<i8>,Vec<i32>) {
 assert!(cols>0 && cols%256==0 && w.len()%cols==0);
 let mut packed=vec![0;w.len()]; let mut factors=vec![0;w.len()/256];
 for (b,chunk) in w.chunks_exact(256).enumerate() {
  for i in 0..128 {packed[b*256+i]=chunk[2*i];packed[b*256+128+i]=chunk[2*i+1];factors[b]+=chunk[2*i] as i32*chunk[2*i+1] as i32;}
 }
 (packed,factors)
}
pub fn activation(q:&QuantizedRows)->(Vec<i16>,Vec<i32>) {
 let mut packed=vec![0;q.values().len()];let mut factors=vec![0;q.values().len()/256];
 for (b,chunk) in q.values().chunks_exact(256).enumerate() {
  for i in 0..128 {packed[b*256+i]=chunk[2*i];packed[b*256+128+i]=chunk[2*i+1];factors[b]+=chunk[2*i] as i32*chunk[2*i+1] as i32;}
 }
 (packed,factors)
}
pub fn project(q:&QuantizedRows,qp:&[i16],qf:&[i32],w:&[i8],wf:&[i32],scales:&[f32],rows:usize)->Vec<f32> {
 let cols=q.cols();assert!(rows>0 && rows*cols<=30_000_000 && q.rows()*rows<=900_000 && scales.iter().all(|v|v.is_finite()&&*v>0.));assert_eq!(w.len(),rows*cols);assert_eq!(wf.len(),rows*cols/256);assert_eq!(scales.len(),rows);
 assert_eq!(qp.len(),q.values().len());assert_eq!(qf.len(),qp.len()/256);assert!(rows%8==0);
 let mut out=vec![0.;q.rows()*rows];
 for r in (0..rows).step_by(8) {
  let mut t=0;let padded=if q.rows()<8 {q.rows()}else{q.rows().div_ceil(8)*8};
  while t<padded {
   macro_rules! tile {($n:literal)=>{if t+$n<=padded {project_tile::<$n>(q,qp,qf,w,wf,scales,rows,t,r,&mut out);t+=$n;continue;}};}
   tile!(64);tile!(32);tile!(16);tile!(8);tile!(4);tile!(2);tile!(1);
  }
 }
 assert!(out.iter().all(|v|v.is_finite()));
 out
}
fn project_tile<const R:usize>(q:&QuantizedRows,qp:&[i16],qf:&[i32],w:&[i8],wf:&[i32],scales:&[f32],rows:usize,t:usize,r:usize,out:&mut[f32]) {
 let cols=q.cols();let mut sums=[[0f32;8];R];
 for b in 0..cols/256 {
  #[cfg(target_arch="wasm32")]
  let dots=unsafe {dot_pair::<R,8>(qp.as_ptr().add(t*cols),w.as_ptr().add(r*cols),cols,b*256,qf.as_ptr().add(t*cols/256),wf.as_ptr().add(r*cols/256),b)};
  #[cfg(not(target_arch="wasm32"))]
  let dots={let mut d=[[0i32;8];R];for i in 0..R {for j in 0..8 {let a=(t+i)*cols+b*256;let z=(r+j)*cols+b*256;
    for k in 0..128 {d[i][j]+=(qp[a+k] as i32+w[z+128+k] as i32)*(qp[a+128+k] as i32+w[z+k] as i32);}
    d[i][j]-=qf[(t+i)*cols/256+b]+wf[(r+j)*cols/256+b];}}d};
  #[cfg(target_arch="wasm32")]
  unsafe {scale_tile::<R,8>(&dots,&mut sums,q.scales().as_ptr().add(t*(cols/256)+b),cols/256,scales.as_ptr().add(r));}
  #[cfg(not(target_arch="wasm32"))]
  for i in 0..R {for j in 0..8 {sums[i][j]+=(dots[i][j] as f32*q.scales()[(t+i)*cols/256+b])*scales[r+j];}}
 }
 for i in 0..R {if t+i<q.rows(){out[(t+i)*rows+r..(t+i)*rows+r+8].copy_from_slice(&sums[i]);}}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn dot_pair<const R: usize, const C: usize>(
    q: *const i16,
    w: *const i8,
    cols: usize,
    start: usize,
    qf: *const i32, wf: *const i32, block: usize,
) -> [[i32; C]; R] {
    assert!(R <= 64);
    const { assert!(C <= 16 && C % 4 == 0); }
    use core::arch::wasm32::*;
    let mut out = [[0i32; C]; R];
    let qp: [*const i16; R] = core::array::from_fn(|i| q.add(i * cols + start));
    let wp: [*const i8; C] = core::array::from_fn(|j| w.add(j * cols + start));
    for c in (0..256).step_by(256) {
        let weights: [[v128; 32]; C] = core::array::from_fn(|j| {
            core::array::from_fn(|g| {
                i16x8_extend_low_i8x16(v128_load64_zero(wp[j].add(c + g * 8).cast()))
            })
        });
        macro_rules! dot { ($input:ident,$j:expr) => { i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8(i16x8_add($input[0], weights[$j][16]), i16x8_add($input[16], weights[$j][0])), i32x4_dot_i16x8(i16x8_add($input[1], weights[$j][17]), i16x8_add($input[17], weights[$j][1]))), i32x4_add(i32x4_dot_i16x8(i16x8_add($input[2], weights[$j][18]), i16x8_add($input[18], weights[$j][2])), i32x4_dot_i16x8(i16x8_add($input[3], weights[$j][19]), i16x8_add($input[19], weights[$j][3])))), i32x4_add(i32x4_add(i32x4_dot_i16x8(i16x8_add($input[4], weights[$j][20]), i16x8_add($input[20], weights[$j][4])), i32x4_dot_i16x8(i16x8_add($input[5], weights[$j][21]), i16x8_add($input[21], weights[$j][5]))), i32x4_add(i32x4_dot_i16x8(i16x8_add($input[6], weights[$j][22]), i16x8_add($input[22], weights[$j][6])), i32x4_dot_i16x8(i16x8_add($input[7], weights[$j][23]), i16x8_add($input[23], weights[$j][7]))))), i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8(i16x8_add($input[8], weights[$j][24]), i16x8_add($input[24], weights[$j][8])), i32x4_dot_i16x8(i16x8_add($input[9], weights[$j][25]), i16x8_add($input[25], weights[$j][9]))), i32x4_add(i32x4_dot_i16x8(i16x8_add($input[10], weights[$j][26]), i16x8_add($input[26], weights[$j][10])), i32x4_dot_i16x8(i16x8_add($input[11], weights[$j][27]), i16x8_add($input[27], weights[$j][11])))), i32x4_add(i32x4_add(i32x4_dot_i16x8(i16x8_add($input[12], weights[$j][28]), i16x8_add($input[28], weights[$j][12])), i32x4_dot_i16x8(i16x8_add($input[13], weights[$j][29]), i16x8_add($input[29], weights[$j][13]))), i32x4_add(i32x4_dot_i16x8(i16x8_add($input[14], weights[$j][30]), i16x8_add($input[30], weights[$j][14])), i32x4_dot_i16x8(i16x8_add($input[15], weights[$j][31]), i16x8_add($input[31], weights[$j][15])))))) }; }
        macro_rules! group {
            ($i:literal,$input:ident,$j:literal) => {
                if C >= $j + 4 {
                    let a0 = dot!($input, $j);
                    let a1 = dot!($input, { $j + 1 });
                    let a2 = dot!($input, { $j + 2 });
                    let a3 = dot!($input, { $j + 3 });
                    let a = i32x4_add(
                        i32x4_shuffle::<0, 1, 4, 5>(a0, a1),
                        i32x4_shuffle::<2, 3, 6, 7>(a0, a1),
                    );
                    let b = i32x4_add(
                        i32x4_shuffle::<0, 1, 4, 5>(a2, a3),
                        i32x4_shuffle::<2, 3, 6, 7>(a2, a3),
                    );
                    let total = i32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(a, b),
                        i32x4_shuffle::<1, 3, 5, 7>(a, b),
                    );
                    // Weight factors have a row stride; gather the four fixed corrections.
                    let wf4 = i32x4(*wf.add(($j)*(cols/256)+block), *wf.add(($j+1)*(cols/256)+block), *wf.add(($j+2)*(cols/256)+block), *wf.add(($j+3)*(cols/256)+block));
                    let corrected = i32x4_sub(i32x4_sub(total, wf4), i32x4_splat(*qf.add($i*(cols/256)+block)));
                    v128_store(out[$i].as_mut_ptr().add($j).cast(), corrected);
                }
            };
        }
        macro_rules! columns {
            ($i:literal,$input:ident;$($unused:literal),*) => {
                group!($i, $input, 0);
                group!($i, $input, 4);
                group!($i, $input, 8);
                group!($i, $input, 12);
            };
        }
        macro_rules! rows {($($i:literal),*)=>{$(if R>$i {
            let input:[v128;32]=core::array::from_fn(|g| v128_load(qp[$i].add(c+g*8).cast()));
            columns!($i,input;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        })*};}
        rows!(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        );
    }
    out
}


#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn scale_tile<const R: usize, const C: usize>(
    dots: &[[i32; C]; R],
    sums: &mut [[f32; C]; R],
    sx: *const f32,
    stride: usize,
    sw: *const f32,
) {
    use core::arch::wasm32::*;
    for i in 0..R {
        let activation = f32x4_splat(*sx.add(i * stride));
        for j in (0..C).step_by(4) {
            let d = v128_load(dots.as_ptr().cast::<i32>().add(i * C + j).cast());
            let w = v128_load(sw.add(j).cast());
            let value = f32x4_mul(f32x4_mul(f32x4_convert_i32x4(d), activation), w);
            let p = sums.as_mut_ptr().cast::<f32>().add(i * C + j);
            v128_store(p.cast(), f32x4_add(v128_load(p.cast()), value));
        }
    }
}
