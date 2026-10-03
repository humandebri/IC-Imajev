//! Generated tile. Validated constructors bound every pointer span.
//! Wrapping address operations only remove redundant checks; no valid span wraps.
use super::{Prepared,Operands};
use imajev_runtime::int8_kernel::QuantizedRows;
#[target_feature(enable="simd128")]
pub(super) unsafe fn evaluate<const R:usize>(w:&Prepared,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize,t:usize,r:usize,out:&mut[f32]) {
 use core::arch::wasm32::*;
 const{assert!(R<=32);}
 let stride=w.cols/2;let blocks=w.cols/256;let zero=f32x4_splat(0.);
 let mut sums=[[[[zero;16];2];2];R];
 let ap:[*const i16;7]=core::array::from_fn(|m:usize|a.data.as_ptr().add(m.wrapping_mul(a.pairs).wrapping_mul(stride).wrapping_add((t/2).wrapping_mul(stride))));
 let wp:[*const i16;7]=core::array::from_fn(|m:usize|w.data.as_ptr().add(m.wrapping_mul(w.rows/2).wrapping_mul(stride).wrapping_add((r/2).wrapping_mul(stride))));
 let sp=q.scales().as_ptr().add(t.wrapping_mul(blocks));
 let sw0=v128_load(sw.as_ptr().add(r+0).cast());
 let sw1=v128_load(sw.as_ptr().add(r+4).cast());
 let sw2=v128_load(sw.as_ptr().add(r+8).cast());
 let sw3=v128_load(sw.as_ptr().add(r+12).cast());
 let sw4=v128_load(sw.as_ptr().add(r+16).cast());
 let sw5=v128_load(sw.as_ptr().add(r+20).cast());
 let sw6=v128_load(sw.as_ptr().add(r+24).cast());
 let sw7=v128_load(sw.as_ptr().add(r+28).cast());
 let sw8=v128_load(sw.as_ptr().add(r+32).cast());
 let sw9=v128_load(sw.as_ptr().add(r+36).cast());
 let sw10=v128_load(sw.as_ptr().add(r+40).cast());
 let sw11=v128_load(sw.as_ptr().add(r+44).cast());
 let sw12=v128_load(sw.as_ptr().add(r+48).cast());
 let sw13=v128_load(sw.as_ptr().add(r+52).cast());
 let sw14=v128_load(sw.as_ptr().add(r+56).cast());
 let sw15=v128_load(sw.as_ptr().add(r+60).cast());
 let sw16=v128_load(sw.as_ptr().add(r+64).cast());
 let sw17=v128_load(sw.as_ptr().add(r+68).cast());
 let sw18=v128_load(sw.as_ptr().add(r+72).cast());
 let sw19=v128_load(sw.as_ptr().add(r+76).cast());
 let sw20=v128_load(sw.as_ptr().add(r+80).cast());
 let sw21=v128_load(sw.as_ptr().add(r+84).cast());
 let sw22=v128_load(sw.as_ptr().add(r+88).cast());
 let sw23=v128_load(sw.as_ptr().add(r+92).cast());
 let sw24=v128_load(sw.as_ptr().add(r+96).cast());
 let sw25=v128_load(sw.as_ptr().add(r+100).cast());
 let sw26=v128_load(sw.as_ptr().add(r+104).cast());
 let sw27=v128_load(sw.as_ptr().add(r+108).cast());
 let sw28=v128_load(sw.as_ptr().add(r+112).cast());
 let sw29=v128_load(sw.as_ptr().add(r+116).cast());
 let sw30=v128_load(sw.as_ptr().add(r+120).cast());
 let sw31=v128_load(sw.as_ptr().add(r+124).cast());
 let even0=i32x4_shuffle::<0,2,4,6>(sw0,sw1);let odd0=i32x4_shuffle::<1,3,5,7>(sw0,sw1);
 let even1=i32x4_shuffle::<0,2,4,6>(sw2,sw3);let odd1=i32x4_shuffle::<1,3,5,7>(sw2,sw3);
 let even2=i32x4_shuffle::<0,2,4,6>(sw4,sw5);let odd2=i32x4_shuffle::<1,3,5,7>(sw4,sw5);
 let even3=i32x4_shuffle::<0,2,4,6>(sw6,sw7);let odd3=i32x4_shuffle::<1,3,5,7>(sw6,sw7);
 let even4=i32x4_shuffle::<0,2,4,6>(sw8,sw9);let odd4=i32x4_shuffle::<1,3,5,7>(sw8,sw9);
 let even5=i32x4_shuffle::<0,2,4,6>(sw10,sw11);let odd5=i32x4_shuffle::<1,3,5,7>(sw10,sw11);
 let even6=i32x4_shuffle::<0,2,4,6>(sw12,sw13);let odd6=i32x4_shuffle::<1,3,5,7>(sw12,sw13);
 let even7=i32x4_shuffle::<0,2,4,6>(sw14,sw15);let odd7=i32x4_shuffle::<1,3,5,7>(sw14,sw15);
 let even8=i32x4_shuffle::<0,2,4,6>(sw16,sw17);let odd8=i32x4_shuffle::<1,3,5,7>(sw16,sw17);
 let even9=i32x4_shuffle::<0,2,4,6>(sw18,sw19);let odd9=i32x4_shuffle::<1,3,5,7>(sw18,sw19);
 let even10=i32x4_shuffle::<0,2,4,6>(sw20,sw21);let odd10=i32x4_shuffle::<1,3,5,7>(sw20,sw21);
 let even11=i32x4_shuffle::<0,2,4,6>(sw22,sw23);let odd11=i32x4_shuffle::<1,3,5,7>(sw22,sw23);
 let even12=i32x4_shuffle::<0,2,4,6>(sw24,sw25);let odd12=i32x4_shuffle::<1,3,5,7>(sw24,sw25);
 let even13=i32x4_shuffle::<0,2,4,6>(sw26,sw27);let odd13=i32x4_shuffle::<1,3,5,7>(sw26,sw27);
 let even14=i32x4_shuffle::<0,2,4,6>(sw28,sw29);let odd14=i32x4_shuffle::<1,3,5,7>(sw28,sw29);
 let even15=i32x4_shuffle::<0,2,4,6>(sw30,sw31);let odd15=i32x4_shuffle::<1,3,5,7>(sw30,sw31);
 for block in 0..blocks {
  let start=block.wrapping_mul(128);
  let products=[
   crate::kernel::dot_tile::<R>(ap[0].add(start),wp[0].add(start),stride),
   crate::kernel::dot_tile::<R>(ap[1].add(start),wp[1].add(start),stride),
   crate::kernel::dot_tile::<R>(ap[2].add(start),wp[2].add(start),stride),
   crate::kernel::dot_tile::<R>(ap[3].add(start),wp[3].add(start),stride),
   crate::kernel::dot_tile::<R>(ap[4].add(start),wp[4].add(start),stride),
   crate::kernel::dot_tile::<R>(ap[5].add(start),wp[5].add(start),stride),
   crate::kernel::dot_tile::<R>(ap[6].add(start),wp[6].add(start),stride),
  ];
  macro_rules! row {($i:literal)=>{if R>$i {
   let sx0=f32x4_splat(*sp.add(($i*2usize).wrapping_mul(blocks).wrapping_add(block)));
   let sx1=f32x4_splat(*sp.add(($i*2usize+1).wrapping_mul(blocks).wrapping_add(block)));
   {
    let p0=v128_load(products[0][$i].as_ptr().add(0).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(0).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(0).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(0).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(0).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(0).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(0).cast());
    sums[$i][0][0][0]=f32x4_add(sums[$i][0][0][0],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even0));
    sums[$i][0][1][0]=f32x4_add(sums[$i][0][1][0],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd0));
    sums[$i][1][0][0]=f32x4_add(sums[$i][1][0][0],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even0));
    sums[$i][1][1][0]=f32x4_add(sums[$i][1][1][0],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd0));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(4).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(4).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(4).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(4).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(4).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(4).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(4).cast());
    sums[$i][0][0][1]=f32x4_add(sums[$i][0][0][1],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even1));
    sums[$i][0][1][1]=f32x4_add(sums[$i][0][1][1],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd1));
    sums[$i][1][0][1]=f32x4_add(sums[$i][1][0][1],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even1));
    sums[$i][1][1][1]=f32x4_add(sums[$i][1][1][1],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd1));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(8).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(8).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(8).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(8).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(8).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(8).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(8).cast());
    sums[$i][0][0][2]=f32x4_add(sums[$i][0][0][2],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even2));
    sums[$i][0][1][2]=f32x4_add(sums[$i][0][1][2],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd2));
    sums[$i][1][0][2]=f32x4_add(sums[$i][1][0][2],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even2));
    sums[$i][1][1][2]=f32x4_add(sums[$i][1][1][2],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd2));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(12).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(12).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(12).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(12).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(12).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(12).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(12).cast());
    sums[$i][0][0][3]=f32x4_add(sums[$i][0][0][3],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even3));
    sums[$i][0][1][3]=f32x4_add(sums[$i][0][1][3],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd3));
    sums[$i][1][0][3]=f32x4_add(sums[$i][1][0][3],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even3));
    sums[$i][1][1][3]=f32x4_add(sums[$i][1][1][3],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd3));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(16).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(16).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(16).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(16).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(16).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(16).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(16).cast());
    sums[$i][0][0][4]=f32x4_add(sums[$i][0][0][4],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even4));
    sums[$i][0][1][4]=f32x4_add(sums[$i][0][1][4],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd4));
    sums[$i][1][0][4]=f32x4_add(sums[$i][1][0][4],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even4));
    sums[$i][1][1][4]=f32x4_add(sums[$i][1][1][4],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd4));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(20).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(20).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(20).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(20).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(20).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(20).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(20).cast());
    sums[$i][0][0][5]=f32x4_add(sums[$i][0][0][5],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even5));
    sums[$i][0][1][5]=f32x4_add(sums[$i][0][1][5],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd5));
    sums[$i][1][0][5]=f32x4_add(sums[$i][1][0][5],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even5));
    sums[$i][1][1][5]=f32x4_add(sums[$i][1][1][5],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd5));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(24).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(24).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(24).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(24).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(24).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(24).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(24).cast());
    sums[$i][0][0][6]=f32x4_add(sums[$i][0][0][6],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even6));
    sums[$i][0][1][6]=f32x4_add(sums[$i][0][1][6],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd6));
    sums[$i][1][0][6]=f32x4_add(sums[$i][1][0][6],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even6));
    sums[$i][1][1][6]=f32x4_add(sums[$i][1][1][6],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd6));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(28).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(28).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(28).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(28).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(28).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(28).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(28).cast());
    sums[$i][0][0][7]=f32x4_add(sums[$i][0][0][7],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even7));
    sums[$i][0][1][7]=f32x4_add(sums[$i][0][1][7],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd7));
    sums[$i][1][0][7]=f32x4_add(sums[$i][1][0][7],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even7));
    sums[$i][1][1][7]=f32x4_add(sums[$i][1][1][7],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd7));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(32).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(32).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(32).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(32).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(32).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(32).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(32).cast());
    sums[$i][0][0][8]=f32x4_add(sums[$i][0][0][8],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even8));
    sums[$i][0][1][8]=f32x4_add(sums[$i][0][1][8],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd8));
    sums[$i][1][0][8]=f32x4_add(sums[$i][1][0][8],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even8));
    sums[$i][1][1][8]=f32x4_add(sums[$i][1][1][8],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd8));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(36).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(36).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(36).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(36).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(36).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(36).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(36).cast());
    sums[$i][0][0][9]=f32x4_add(sums[$i][0][0][9],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even9));
    sums[$i][0][1][9]=f32x4_add(sums[$i][0][1][9],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd9));
    sums[$i][1][0][9]=f32x4_add(sums[$i][1][0][9],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even9));
    sums[$i][1][1][9]=f32x4_add(sums[$i][1][1][9],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd9));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(40).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(40).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(40).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(40).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(40).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(40).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(40).cast());
    sums[$i][0][0][10]=f32x4_add(sums[$i][0][0][10],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even10));
    sums[$i][0][1][10]=f32x4_add(sums[$i][0][1][10],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd10));
    sums[$i][1][0][10]=f32x4_add(sums[$i][1][0][10],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even10));
    sums[$i][1][1][10]=f32x4_add(sums[$i][1][1][10],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd10));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(44).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(44).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(44).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(44).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(44).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(44).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(44).cast());
    sums[$i][0][0][11]=f32x4_add(sums[$i][0][0][11],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even11));
    sums[$i][0][1][11]=f32x4_add(sums[$i][0][1][11],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd11));
    sums[$i][1][0][11]=f32x4_add(sums[$i][1][0][11],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even11));
    sums[$i][1][1][11]=f32x4_add(sums[$i][1][1][11],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd11));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(48).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(48).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(48).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(48).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(48).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(48).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(48).cast());
    sums[$i][0][0][12]=f32x4_add(sums[$i][0][0][12],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even12));
    sums[$i][0][1][12]=f32x4_add(sums[$i][0][1][12],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd12));
    sums[$i][1][0][12]=f32x4_add(sums[$i][1][0][12],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even12));
    sums[$i][1][1][12]=f32x4_add(sums[$i][1][1][12],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd12));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(52).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(52).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(52).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(52).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(52).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(52).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(52).cast());
    sums[$i][0][0][13]=f32x4_add(sums[$i][0][0][13],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even13));
    sums[$i][0][1][13]=f32x4_add(sums[$i][0][1][13],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd13));
    sums[$i][1][0][13]=f32x4_add(sums[$i][1][0][13],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even13));
    sums[$i][1][1][13]=f32x4_add(sums[$i][1][1][13],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd13));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(56).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(56).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(56).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(56).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(56).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(56).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(56).cast());
    sums[$i][0][0][14]=f32x4_add(sums[$i][0][0][14],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even14));
    sums[$i][0][1][14]=f32x4_add(sums[$i][0][1][14],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd14));
    sums[$i][1][0][14]=f32x4_add(sums[$i][1][0][14],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even14));
    sums[$i][1][1][14]=f32x4_add(sums[$i][1][1][14],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd14));
   }
   {
    let p0=v128_load(products[0][$i].as_ptr().add(60).cast());
    let p1=v128_load(products[1][$i].as_ptr().add(60).cast());
    let p2=v128_load(products[2][$i].as_ptr().add(60).cast());
    let p3=v128_load(products[3][$i].as_ptr().add(60).cast());
    let p4=v128_load(products[4][$i].as_ptr().add(60).cast());
    let p5=v128_load(products[5][$i].as_ptr().add(60).cast());
    let p6=v128_load(products[6][$i].as_ptr().add(60).cast());
    sums[$i][0][0][15]=f32x4_add(sums[$i][0][0][15],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_sub(i32x4_add(p0,p3),p4),p6)),sx0),even15));
    sums[$i][0][1][15]=f32x4_add(sums[$i][0][1][15],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p2,p4)),sx0),odd15));
    sums[$i][1][0][15]=f32x4_add(sums[$i][1][0][15],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(p1,p3)),sx1),even15));
    sums[$i][1][1][15]=f32x4_add(sums[$i][1][1][15],f32x4_mul(f32x4_mul(f32x4_convert_i32x4(i32x4_add(i32x4_add(i32x4_sub(p0,p1),p2),p5)),sx1),odd15));
   }
  }}}
  row!(0);
  row!(1);
  row!(2);
  row!(3);
  row!(4);
  row!(5);
  row!(6);
  row!(7);
  row!(8);
  row!(9);
  row!(10);
  row!(11);
  row!(12);
  row!(13);
  row!(14);
  row!(15);
  row!(16);
  row!(17);
  row!(18);
  row!(19);
  row!(20);
  row!(21);
  row!(22);
  row!(23);
  row!(24);
  row!(25);
  row!(26);
  row!(27);
  row!(28);
  row!(29);
  row!(30);
  row!(31);
 }
 macro_rules! store {($i:literal)=>{if R>$i {
  {let token=t.wrapping_add($i*2usize+0);if token<q.rows() {let dst=out.as_mut_ptr().add(token.wrapping_mul(rows).wrapping_add(r));
   {let even=sums[$i][0][0][0];let odd=sums[$i][0][1][0];
   v128_store(dst.add(0).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(4).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][1];let odd=sums[$i][0][1][1];
   v128_store(dst.add(8).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(12).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][2];let odd=sums[$i][0][1][2];
   v128_store(dst.add(16).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(20).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][3];let odd=sums[$i][0][1][3];
   v128_store(dst.add(24).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(28).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][4];let odd=sums[$i][0][1][4];
   v128_store(dst.add(32).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(36).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][5];let odd=sums[$i][0][1][5];
   v128_store(dst.add(40).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(44).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][6];let odd=sums[$i][0][1][6];
   v128_store(dst.add(48).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(52).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][7];let odd=sums[$i][0][1][7];
   v128_store(dst.add(56).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(60).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][8];let odd=sums[$i][0][1][8];
   v128_store(dst.add(64).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(68).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][9];let odd=sums[$i][0][1][9];
   v128_store(dst.add(72).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(76).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][10];let odd=sums[$i][0][1][10];
   v128_store(dst.add(80).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(84).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][11];let odd=sums[$i][0][1][11];
   v128_store(dst.add(88).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(92).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][12];let odd=sums[$i][0][1][12];
   v128_store(dst.add(96).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(100).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][13];let odd=sums[$i][0][1][13];
   v128_store(dst.add(104).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(108).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][14];let odd=sums[$i][0][1][14];
   v128_store(dst.add(112).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(116).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][0][0][15];let odd=sums[$i][0][1][15];
   v128_store(dst.add(120).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(124).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
  } }
  {let token=t.wrapping_add($i*2usize+1);if token<q.rows() {let dst=out.as_mut_ptr().add(token.wrapping_mul(rows).wrapping_add(r));
   {let even=sums[$i][1][0][0];let odd=sums[$i][1][1][0];
   v128_store(dst.add(0).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(4).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][1];let odd=sums[$i][1][1][1];
   v128_store(dst.add(8).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(12).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][2];let odd=sums[$i][1][1][2];
   v128_store(dst.add(16).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(20).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][3];let odd=sums[$i][1][1][3];
   v128_store(dst.add(24).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(28).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][4];let odd=sums[$i][1][1][4];
   v128_store(dst.add(32).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(36).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][5];let odd=sums[$i][1][1][5];
   v128_store(dst.add(40).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(44).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][6];let odd=sums[$i][1][1][6];
   v128_store(dst.add(48).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(52).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][7];let odd=sums[$i][1][1][7];
   v128_store(dst.add(56).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(60).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][8];let odd=sums[$i][1][1][8];
   v128_store(dst.add(64).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(68).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][9];let odd=sums[$i][1][1][9];
   v128_store(dst.add(72).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(76).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][10];let odd=sums[$i][1][1][10];
   v128_store(dst.add(80).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(84).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][11];let odd=sums[$i][1][1][11];
   v128_store(dst.add(88).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(92).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][12];let odd=sums[$i][1][1][12];
   v128_store(dst.add(96).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(100).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][13];let odd=sums[$i][1][1][13];
   v128_store(dst.add(104).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(108).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][14];let odd=sums[$i][1][1][14];
   v128_store(dst.add(112).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(116).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
   {let even=sums[$i][1][0][15];let odd=sums[$i][1][1][15];
   v128_store(dst.add(120).cast(),i32x4_shuffle::<0,4,1,5>(even,odd));
   v128_store(dst.add(124).cast(),i32x4_shuffle::<2,6,3,7>(even,odd));}
  } }
 }}}
 store!(0);
 store!(1);
 store!(2);
 store!(3);
 store!(4);
 store!(5);
 store!(6);
 store!(7);
 store!(8);
 store!(9);
 store!(10);
 store!(11);
 store!(12);
 store!(13);
 store!(14);
 store!(15);
 store!(16);
 store!(17);
 store!(18);
 store!(19);
 store!(20);
 store!(21);
 store!(22);
 store!(23);
 store!(24);
 store!(25);
 store!(26);
 store!(27);
 store!(28);
 store!(29);
 store!(30);
 store!(31);
}
