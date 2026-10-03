//! Deterministic source from scripts/generate_linear_pair.py.
//! Address spans are validated by the private caller; no valid index wraps.
//! Q and each weight pair occupy eight lanes per K4 group.
//! Integer associativity is exact: |block sum| <= 256*127*128 < 2^31.
#[target_feature(enable="simd128")]
pub(crate) unsafe fn accumulate<const LINEAR:bool>(q:*const i16,w:*const i8,cols:usize,start:usize,sx:*const f32,stride:usize,sw:*const f32,sums:&mut[[f32;32]]) {
debug_assert!(sums.len()<=132);
use core::arch::wasm32::*;
let ws:[v128;8]=core::array::from_fn(|j|v128_load(sw.add(j*4).cast()));
let wp:[*const i8;16]=core::array::from_fn(|j:usize|w.add(j.wrapping_mul(cols).wrapping_mul(2).wrapping_add(start.wrapping_mul(2))));
let mut weights=[[i32x4_splat(0);64];16];
macro_rules! pair {($j:literal)=>{weights[$j]=[
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(0).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(8).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(16).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(24).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(32).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(40).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(48).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(56).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(64).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(72).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(80).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(88).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(96).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(104).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(112).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(120).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(128).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(136).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(144).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(152).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(160).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(168).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(176).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(184).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(192).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(200).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(208).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(216).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(224).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(232).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(240).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(248).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(256).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(264).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(272).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(280).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(288).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(296).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(304).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(312).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(320).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(328).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(336).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(344).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(352).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(360).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(368).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(376).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(384).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(392).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(400).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(408).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(416).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(424).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(432).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(440).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(448).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(456).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(464).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(472).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(480).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(488).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(496).cast())),
i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(504).cast()))
];};}
pair!(0);pair!(1);pair!(2);pair!(3);pair!(4);pair!(5);pair!(6);pair!(7);pair!(8);pair!(9);pair!(10);pair!(11);pair!(12);pair!(13);pair!(14);pair!(15);
macro_rules! dot {($input:ident,$j:expr)=>{{if LINEAR {i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[0],weights[$j][0]),i32x4_dot_i16x8($input[1],weights[$j][1])),i32x4_dot_i16x8($input[2],weights[$j][2])),i32x4_dot_i16x8($input[3],weights[$j][3])),i32x4_dot_i16x8($input[4],weights[$j][4])),i32x4_dot_i16x8($input[5],weights[$j][5])),i32x4_dot_i16x8($input[6],weights[$j][6])),i32x4_dot_i16x8($input[7],weights[$j][7])),i32x4_dot_i16x8($input[8],weights[$j][8])),i32x4_dot_i16x8($input[9],weights[$j][9])),i32x4_dot_i16x8($input[10],weights[$j][10])),i32x4_dot_i16x8($input[11],weights[$j][11])),i32x4_dot_i16x8($input[12],weights[$j][12])),i32x4_dot_i16x8($input[13],weights[$j][13])),i32x4_dot_i16x8($input[14],weights[$j][14])),i32x4_dot_i16x8($input[15],weights[$j][15])),i32x4_dot_i16x8($input[16],weights[$j][16])),i32x4_dot_i16x8($input[17],weights[$j][17])),i32x4_dot_i16x8($input[18],weights[$j][18])),i32x4_dot_i16x8($input[19],weights[$j][19])),i32x4_dot_i16x8($input[20],weights[$j][20])),i32x4_dot_i16x8($input[21],weights[$j][21])),i32x4_dot_i16x8($input[22],weights[$j][22])),i32x4_dot_i16x8($input[23],weights[$j][23])),i32x4_dot_i16x8($input[24],weights[$j][24])),i32x4_dot_i16x8($input[25],weights[$j][25])),i32x4_dot_i16x8($input[26],weights[$j][26])),i32x4_dot_i16x8($input[27],weights[$j][27])),i32x4_dot_i16x8($input[28],weights[$j][28])),i32x4_dot_i16x8($input[29],weights[$j][29])),i32x4_dot_i16x8($input[30],weights[$j][30])),i32x4_dot_i16x8($input[31],weights[$j][31])),i32x4_dot_i16x8($input[32],weights[$j][32])),i32x4_dot_i16x8($input[33],weights[$j][33])),i32x4_dot_i16x8($input[34],weights[$j][34])),i32x4_dot_i16x8($input[35],weights[$j][35])),i32x4_dot_i16x8($input[36],weights[$j][36])),i32x4_dot_i16x8($input[37],weights[$j][37])),i32x4_dot_i16x8($input[38],weights[$j][38])),i32x4_dot_i16x8($input[39],weights[$j][39])),i32x4_dot_i16x8($input[40],weights[$j][40])),i32x4_dot_i16x8($input[41],weights[$j][41])),i32x4_dot_i16x8($input[42],weights[$j][42])),i32x4_dot_i16x8($input[43],weights[$j][43])),i32x4_dot_i16x8($input[44],weights[$j][44])),i32x4_dot_i16x8($input[45],weights[$j][45])),i32x4_dot_i16x8($input[46],weights[$j][46])),i32x4_dot_i16x8($input[47],weights[$j][47])),i32x4_dot_i16x8($input[48],weights[$j][48])),i32x4_dot_i16x8($input[49],weights[$j][49])),i32x4_dot_i16x8($input[50],weights[$j][50])),i32x4_dot_i16x8($input[51],weights[$j][51])),i32x4_dot_i16x8($input[52],weights[$j][52])),i32x4_dot_i16x8($input[53],weights[$j][53])),i32x4_dot_i16x8($input[54],weights[$j][54])),i32x4_dot_i16x8($input[55],weights[$j][55])),i32x4_dot_i16x8($input[56],weights[$j][56])),i32x4_dot_i16x8($input[57],weights[$j][57])),i32x4_dot_i16x8($input[58],weights[$j][58])),i32x4_dot_i16x8($input[59],weights[$j][59])),i32x4_dot_i16x8($input[60],weights[$j][60])),i32x4_dot_i16x8($input[61],weights[$j][61])),i32x4_dot_i16x8($input[62],weights[$j][62])),i32x4_dot_i16x8($input[63],weights[$j][63]))}else{i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[0],weights[$j][0]),i32x4_dot_i16x8($input[1],weights[$j][1])),i32x4_add(i32x4_dot_i16x8($input[2],weights[$j][2]),i32x4_dot_i16x8($input[3],weights[$j][3]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[4],weights[$j][4]),i32x4_dot_i16x8($input[5],weights[$j][5])),i32x4_add(i32x4_dot_i16x8($input[6],weights[$j][6]),i32x4_dot_i16x8($input[7],weights[$j][7])))),i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[8],weights[$j][8]),i32x4_dot_i16x8($input[9],weights[$j][9])),i32x4_add(i32x4_dot_i16x8($input[10],weights[$j][10]),i32x4_dot_i16x8($input[11],weights[$j][11]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[12],weights[$j][12]),i32x4_dot_i16x8($input[13],weights[$j][13])),i32x4_add(i32x4_dot_i16x8($input[14],weights[$j][14]),i32x4_dot_i16x8($input[15],weights[$j][15]))))),i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[16],weights[$j][16]),i32x4_dot_i16x8($input[17],weights[$j][17])),i32x4_add(i32x4_dot_i16x8($input[18],weights[$j][18]),i32x4_dot_i16x8($input[19],weights[$j][19]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[20],weights[$j][20]),i32x4_dot_i16x8($input[21],weights[$j][21])),i32x4_add(i32x4_dot_i16x8($input[22],weights[$j][22]),i32x4_dot_i16x8($input[23],weights[$j][23])))),i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[24],weights[$j][24]),i32x4_dot_i16x8($input[25],weights[$j][25])),i32x4_add(i32x4_dot_i16x8($input[26],weights[$j][26]),i32x4_dot_i16x8($input[27],weights[$j][27]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[28],weights[$j][28]),i32x4_dot_i16x8($input[29],weights[$j][29])),i32x4_add(i32x4_dot_i16x8($input[30],weights[$j][30]),i32x4_dot_i16x8($input[31],weights[$j][31])))))),i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[32],weights[$j][32]),i32x4_dot_i16x8($input[33],weights[$j][33])),i32x4_add(i32x4_dot_i16x8($input[34],weights[$j][34]),i32x4_dot_i16x8($input[35],weights[$j][35]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[36],weights[$j][36]),i32x4_dot_i16x8($input[37],weights[$j][37])),i32x4_add(i32x4_dot_i16x8($input[38],weights[$j][38]),i32x4_dot_i16x8($input[39],weights[$j][39])))),i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[40],weights[$j][40]),i32x4_dot_i16x8($input[41],weights[$j][41])),i32x4_add(i32x4_dot_i16x8($input[42],weights[$j][42]),i32x4_dot_i16x8($input[43],weights[$j][43]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[44],weights[$j][44]),i32x4_dot_i16x8($input[45],weights[$j][45])),i32x4_add(i32x4_dot_i16x8($input[46],weights[$j][46]),i32x4_dot_i16x8($input[47],weights[$j][47]))))),i32x4_add(i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[48],weights[$j][48]),i32x4_dot_i16x8($input[49],weights[$j][49])),i32x4_add(i32x4_dot_i16x8($input[50],weights[$j][50]),i32x4_dot_i16x8($input[51],weights[$j][51]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[52],weights[$j][52]),i32x4_dot_i16x8($input[53],weights[$j][53])),i32x4_add(i32x4_dot_i16x8($input[54],weights[$j][54]),i32x4_dot_i16x8($input[55],weights[$j][55])))),i32x4_add(i32x4_add(i32x4_add(i32x4_dot_i16x8($input[56],weights[$j][56]),i32x4_dot_i16x8($input[57],weights[$j][57])),i32x4_add(i32x4_dot_i16x8($input[58],weights[$j][58]),i32x4_dot_i16x8($input[59],weights[$j][59]))),i32x4_add(i32x4_add(i32x4_dot_i16x8($input[60],weights[$j][60]),i32x4_dot_i16x8($input[61],weights[$j][61])),i32x4_add(i32x4_dot_i16x8($input[62],weights[$j][62]),i32x4_dot_i16x8($input[63],weights[$j][63])))))))}}};}
macro_rules! group {($i:expr,$input:ident,$j:literal)=>{{
let a=dot!($input,$j/2);let b=dot!($input,$j/2+1);
let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
let scaled=f32x4_mul(f32x4_mul(f32x4_convert_i32x4(total),f32x4_splat(*sx.add(($i as usize).wrapping_mul(stride)))),ws[$j/4]);
let dst=sums[$i].as_mut_ptr().add($j);v128_store(dst.cast(),f32x4_add(v128_load(dst.cast()),scaled));
}};}
for i in 0..sums.len() {
let row=q.add(i.wrapping_mul(cols).wrapping_add(start).wrapping_mul(2));
let input:[v128;64]=[v128_load(row.add(0).cast()),v128_load(row.add(8).cast()),v128_load(row.add(16).cast()),v128_load(row.add(24).cast()),v128_load(row.add(32).cast()),v128_load(row.add(40).cast()),v128_load(row.add(48).cast()),v128_load(row.add(56).cast()),v128_load(row.add(64).cast()),v128_load(row.add(72).cast()),v128_load(row.add(80).cast()),v128_load(row.add(88).cast()),v128_load(row.add(96).cast()),v128_load(row.add(104).cast()),v128_load(row.add(112).cast()),v128_load(row.add(120).cast()),v128_load(row.add(128).cast()),v128_load(row.add(136).cast()),v128_load(row.add(144).cast()),v128_load(row.add(152).cast()),v128_load(row.add(160).cast()),v128_load(row.add(168).cast()),v128_load(row.add(176).cast()),v128_load(row.add(184).cast()),v128_load(row.add(192).cast()),v128_load(row.add(200).cast()),v128_load(row.add(208).cast()),v128_load(row.add(216).cast()),v128_load(row.add(224).cast()),v128_load(row.add(232).cast()),v128_load(row.add(240).cast()),v128_load(row.add(248).cast()),v128_load(row.add(256).cast()),v128_load(row.add(264).cast()),v128_load(row.add(272).cast()),v128_load(row.add(280).cast()),v128_load(row.add(288).cast()),v128_load(row.add(296).cast()),v128_load(row.add(304).cast()),v128_load(row.add(312).cast()),v128_load(row.add(320).cast()),v128_load(row.add(328).cast()),v128_load(row.add(336).cast()),v128_load(row.add(344).cast()),v128_load(row.add(352).cast()),v128_load(row.add(360).cast()),v128_load(row.add(368).cast()),v128_load(row.add(376).cast()),v128_load(row.add(384).cast()),v128_load(row.add(392).cast()),v128_load(row.add(400).cast()),v128_load(row.add(408).cast()),v128_load(row.add(416).cast()),v128_load(row.add(424).cast()),v128_load(row.add(432).cast()),v128_load(row.add(440).cast()),v128_load(row.add(448).cast()),v128_load(row.add(456).cast()),v128_load(row.add(464).cast()),v128_load(row.add(472).cast()),v128_load(row.add(480).cast()),v128_load(row.add(488).cast()),v128_load(row.add(496).cast()),v128_load(row.add(504).cast())];
group!(i,input,0);group!(i,input,4);group!(i,input,8);group!(i,input,12);group!(i,input,16);group!(i,input,20);group!(i,input,24);group!(i,input,28);
}

}
