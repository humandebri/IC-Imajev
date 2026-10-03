// Generated K64 leaf: 32 row quartets share each input vector.
#[target_feature(enable="simd128")]
pub(crate) unsafe fn dot_tile<const R:usize>(q:*const i16,w:*const i16,stride:usize)->[[i32;32];R] {
 use core::arch::wasm32::*;const{assert!(R<=32);}
 let qp:[*const i16;R]=core::array::from_fn(|i|q.add(i.wrapping_mul(stride)));
 let wp:[*const i16;32]=core::array::from_fn(|i|w.add(i.wrapping_mul(stride)));
 let weights=[
 [v128_load(wp[0].add(0).cast()),v128_load(wp[0].add(8).cast()),v128_load(wp[0].add(16).cast()),v128_load(wp[0].add(24).cast()),v128_load(wp[0].add(32).cast()),v128_load(wp[0].add(40).cast()),v128_load(wp[0].add(48).cast()),v128_load(wp[0].add(56).cast())],
 [v128_load(wp[1].add(0).cast()),v128_load(wp[1].add(8).cast()),v128_load(wp[1].add(16).cast()),v128_load(wp[1].add(24).cast()),v128_load(wp[1].add(32).cast()),v128_load(wp[1].add(40).cast()),v128_load(wp[1].add(48).cast()),v128_load(wp[1].add(56).cast())],
 [v128_load(wp[2].add(0).cast()),v128_load(wp[2].add(8).cast()),v128_load(wp[2].add(16).cast()),v128_load(wp[2].add(24).cast()),v128_load(wp[2].add(32).cast()),v128_load(wp[2].add(40).cast()),v128_load(wp[2].add(48).cast()),v128_load(wp[2].add(56).cast())],
 [v128_load(wp[3].add(0).cast()),v128_load(wp[3].add(8).cast()),v128_load(wp[3].add(16).cast()),v128_load(wp[3].add(24).cast()),v128_load(wp[3].add(32).cast()),v128_load(wp[3].add(40).cast()),v128_load(wp[3].add(48).cast()),v128_load(wp[3].add(56).cast())],
 [v128_load(wp[4].add(0).cast()),v128_load(wp[4].add(8).cast()),v128_load(wp[4].add(16).cast()),v128_load(wp[4].add(24).cast()),v128_load(wp[4].add(32).cast()),v128_load(wp[4].add(40).cast()),v128_load(wp[4].add(48).cast()),v128_load(wp[4].add(56).cast())],
 [v128_load(wp[5].add(0).cast()),v128_load(wp[5].add(8).cast()),v128_load(wp[5].add(16).cast()),v128_load(wp[5].add(24).cast()),v128_load(wp[5].add(32).cast()),v128_load(wp[5].add(40).cast()),v128_load(wp[5].add(48).cast()),v128_load(wp[5].add(56).cast())],
 [v128_load(wp[6].add(0).cast()),v128_load(wp[6].add(8).cast()),v128_load(wp[6].add(16).cast()),v128_load(wp[6].add(24).cast()),v128_load(wp[6].add(32).cast()),v128_load(wp[6].add(40).cast()),v128_load(wp[6].add(48).cast()),v128_load(wp[6].add(56).cast())],
 [v128_load(wp[7].add(0).cast()),v128_load(wp[7].add(8).cast()),v128_load(wp[7].add(16).cast()),v128_load(wp[7].add(24).cast()),v128_load(wp[7].add(32).cast()),v128_load(wp[7].add(40).cast()),v128_load(wp[7].add(48).cast()),v128_load(wp[7].add(56).cast())],
 [v128_load(wp[8].add(0).cast()),v128_load(wp[8].add(8).cast()),v128_load(wp[8].add(16).cast()),v128_load(wp[8].add(24).cast()),v128_load(wp[8].add(32).cast()),v128_load(wp[8].add(40).cast()),v128_load(wp[8].add(48).cast()),v128_load(wp[8].add(56).cast())],
 [v128_load(wp[9].add(0).cast()),v128_load(wp[9].add(8).cast()),v128_load(wp[9].add(16).cast()),v128_load(wp[9].add(24).cast()),v128_load(wp[9].add(32).cast()),v128_load(wp[9].add(40).cast()),v128_load(wp[9].add(48).cast()),v128_load(wp[9].add(56).cast())],
 [v128_load(wp[10].add(0).cast()),v128_load(wp[10].add(8).cast()),v128_load(wp[10].add(16).cast()),v128_load(wp[10].add(24).cast()),v128_load(wp[10].add(32).cast()),v128_load(wp[10].add(40).cast()),v128_load(wp[10].add(48).cast()),v128_load(wp[10].add(56).cast())],
 [v128_load(wp[11].add(0).cast()),v128_load(wp[11].add(8).cast()),v128_load(wp[11].add(16).cast()),v128_load(wp[11].add(24).cast()),v128_load(wp[11].add(32).cast()),v128_load(wp[11].add(40).cast()),v128_load(wp[11].add(48).cast()),v128_load(wp[11].add(56).cast())],
 [v128_load(wp[12].add(0).cast()),v128_load(wp[12].add(8).cast()),v128_load(wp[12].add(16).cast()),v128_load(wp[12].add(24).cast()),v128_load(wp[12].add(32).cast()),v128_load(wp[12].add(40).cast()),v128_load(wp[12].add(48).cast()),v128_load(wp[12].add(56).cast())],
 [v128_load(wp[13].add(0).cast()),v128_load(wp[13].add(8).cast()),v128_load(wp[13].add(16).cast()),v128_load(wp[13].add(24).cast()),v128_load(wp[13].add(32).cast()),v128_load(wp[13].add(40).cast()),v128_load(wp[13].add(48).cast()),v128_load(wp[13].add(56).cast())],
 [v128_load(wp[14].add(0).cast()),v128_load(wp[14].add(8).cast()),v128_load(wp[14].add(16).cast()),v128_load(wp[14].add(24).cast()),v128_load(wp[14].add(32).cast()),v128_load(wp[14].add(40).cast()),v128_load(wp[14].add(48).cast()),v128_load(wp[14].add(56).cast())],
 [v128_load(wp[15].add(0).cast()),v128_load(wp[15].add(8).cast()),v128_load(wp[15].add(16).cast()),v128_load(wp[15].add(24).cast()),v128_load(wp[15].add(32).cast()),v128_load(wp[15].add(40).cast()),v128_load(wp[15].add(48).cast()),v128_load(wp[15].add(56).cast())],
 [v128_load(wp[16].add(0).cast()),v128_load(wp[16].add(8).cast()),v128_load(wp[16].add(16).cast()),v128_load(wp[16].add(24).cast()),v128_load(wp[16].add(32).cast()),v128_load(wp[16].add(40).cast()),v128_load(wp[16].add(48).cast()),v128_load(wp[16].add(56).cast())],
 [v128_load(wp[17].add(0).cast()),v128_load(wp[17].add(8).cast()),v128_load(wp[17].add(16).cast()),v128_load(wp[17].add(24).cast()),v128_load(wp[17].add(32).cast()),v128_load(wp[17].add(40).cast()),v128_load(wp[17].add(48).cast()),v128_load(wp[17].add(56).cast())],
 [v128_load(wp[18].add(0).cast()),v128_load(wp[18].add(8).cast()),v128_load(wp[18].add(16).cast()),v128_load(wp[18].add(24).cast()),v128_load(wp[18].add(32).cast()),v128_load(wp[18].add(40).cast()),v128_load(wp[18].add(48).cast()),v128_load(wp[18].add(56).cast())],
 [v128_load(wp[19].add(0).cast()),v128_load(wp[19].add(8).cast()),v128_load(wp[19].add(16).cast()),v128_load(wp[19].add(24).cast()),v128_load(wp[19].add(32).cast()),v128_load(wp[19].add(40).cast()),v128_load(wp[19].add(48).cast()),v128_load(wp[19].add(56).cast())],
 [v128_load(wp[20].add(0).cast()),v128_load(wp[20].add(8).cast()),v128_load(wp[20].add(16).cast()),v128_load(wp[20].add(24).cast()),v128_load(wp[20].add(32).cast()),v128_load(wp[20].add(40).cast()),v128_load(wp[20].add(48).cast()),v128_load(wp[20].add(56).cast())],
 [v128_load(wp[21].add(0).cast()),v128_load(wp[21].add(8).cast()),v128_load(wp[21].add(16).cast()),v128_load(wp[21].add(24).cast()),v128_load(wp[21].add(32).cast()),v128_load(wp[21].add(40).cast()),v128_load(wp[21].add(48).cast()),v128_load(wp[21].add(56).cast())],
 [v128_load(wp[22].add(0).cast()),v128_load(wp[22].add(8).cast()),v128_load(wp[22].add(16).cast()),v128_load(wp[22].add(24).cast()),v128_load(wp[22].add(32).cast()),v128_load(wp[22].add(40).cast()),v128_load(wp[22].add(48).cast()),v128_load(wp[22].add(56).cast())],
 [v128_load(wp[23].add(0).cast()),v128_load(wp[23].add(8).cast()),v128_load(wp[23].add(16).cast()),v128_load(wp[23].add(24).cast()),v128_load(wp[23].add(32).cast()),v128_load(wp[23].add(40).cast()),v128_load(wp[23].add(48).cast()),v128_load(wp[23].add(56).cast())],
 [v128_load(wp[24].add(0).cast()),v128_load(wp[24].add(8).cast()),v128_load(wp[24].add(16).cast()),v128_load(wp[24].add(24).cast()),v128_load(wp[24].add(32).cast()),v128_load(wp[24].add(40).cast()),v128_load(wp[24].add(48).cast()),v128_load(wp[24].add(56).cast())],
 [v128_load(wp[25].add(0).cast()),v128_load(wp[25].add(8).cast()),v128_load(wp[25].add(16).cast()),v128_load(wp[25].add(24).cast()),v128_load(wp[25].add(32).cast()),v128_load(wp[25].add(40).cast()),v128_load(wp[25].add(48).cast()),v128_load(wp[25].add(56).cast())],
 [v128_load(wp[26].add(0).cast()),v128_load(wp[26].add(8).cast()),v128_load(wp[26].add(16).cast()),v128_load(wp[26].add(24).cast()),v128_load(wp[26].add(32).cast()),v128_load(wp[26].add(40).cast()),v128_load(wp[26].add(48).cast()),v128_load(wp[26].add(56).cast())],
 [v128_load(wp[27].add(0).cast()),v128_load(wp[27].add(8).cast()),v128_load(wp[27].add(16).cast()),v128_load(wp[27].add(24).cast()),v128_load(wp[27].add(32).cast()),v128_load(wp[27].add(40).cast()),v128_load(wp[27].add(48).cast()),v128_load(wp[27].add(56).cast())],
 [v128_load(wp[28].add(0).cast()),v128_load(wp[28].add(8).cast()),v128_load(wp[28].add(16).cast()),v128_load(wp[28].add(24).cast()),v128_load(wp[28].add(32).cast()),v128_load(wp[28].add(40).cast()),v128_load(wp[28].add(48).cast()),v128_load(wp[28].add(56).cast())],
 [v128_load(wp[29].add(0).cast()),v128_load(wp[29].add(8).cast()),v128_load(wp[29].add(16).cast()),v128_load(wp[29].add(24).cast()),v128_load(wp[29].add(32).cast()),v128_load(wp[29].add(40).cast()),v128_load(wp[29].add(48).cast()),v128_load(wp[29].add(56).cast())],
 [v128_load(wp[30].add(0).cast()),v128_load(wp[30].add(8).cast()),v128_load(wp[30].add(16).cast()),v128_load(wp[30].add(24).cast()),v128_load(wp[30].add(32).cast()),v128_load(wp[30].add(40).cast()),v128_load(wp[30].add(48).cast()),v128_load(wp[30].add(56).cast())],
 [v128_load(wp[31].add(0).cast()),v128_load(wp[31].add(8).cast()),v128_load(wp[31].add(16).cast()),v128_load(wp[31].add(24).cast()),v128_load(wp[31].add(32).cast()),v128_load(wp[31].add(40).cast()),v128_load(wp[31].add(48).cast()),v128_load(wp[31].add(56).cast())],
 ];
 let mut out=[[0i32;32];R];
 macro_rules! dot {($input:ident,$j:literal)=>{i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[0],weights[$j][0]),i32x4_dot_i16x8($input[1],weights[$j][1])),i32x4_add(i32x4_dot_i16x8($input[2],weights[$j][2]),i32x4_dot_i16x8($input[3],weights[$j][3]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[4],weights[$j][4]),i32x4_dot_i16x8($input[5],weights[$j][5])),i32x4_add(i32x4_dot_i16x8($input[6],weights[$j][6]),i32x4_dot_i16x8($input[7],weights[$j][7]))))};}
 macro_rules! row {($i:literal)=>{if R>$i {
 let input=[v128_load(qp[$i].add(0).cast()),v128_load(qp[$i].add(8).cast()),v128_load(qp[$i].add(16).cast()),v128_load(qp[$i].add(24).cast()),v128_load(qp[$i].add(32).cast()),v128_load(qp[$i].add(40).cast()),v128_load(qp[$i].add(48).cast()),v128_load(qp[$i].add(56).cast())];
 {let a0=dot!(input,0);let a1=dot!(input,1);let a2=dot!(input,2);let a3=dot!(input,3);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(0).cast(),total);}
 {let a0=dot!(input,4);let a1=dot!(input,5);let a2=dot!(input,6);let a3=dot!(input,7);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(4).cast(),total);}
 {let a0=dot!(input,8);let a1=dot!(input,9);let a2=dot!(input,10);let a3=dot!(input,11);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(8).cast(),total);}
 {let a0=dot!(input,12);let a1=dot!(input,13);let a2=dot!(input,14);let a3=dot!(input,15);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(12).cast(),total);}
 {let a0=dot!(input,16);let a1=dot!(input,17);let a2=dot!(input,18);let a3=dot!(input,19);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(16).cast(),total);}
 {let a0=dot!(input,20);let a1=dot!(input,21);let a2=dot!(input,22);let a3=dot!(input,23);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(20).cast(),total);}
 {let a0=dot!(input,24);let a1=dot!(input,25);let a2=dot!(input,26);let a3=dot!(input,27);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(24).cast(),total);}
 {let a0=dot!(input,28);let a1=dot!(input,29);let a2=dot!(input,30);let a3=dot!(input,31);
 let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
 let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
 let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
 v128_store(out[$i].as_mut_ptr().add(28).cast(),total);}
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
 out
}
