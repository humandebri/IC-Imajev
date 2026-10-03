//! Exact block256 dot followed immediately by the original F32 scale/sum.
//! Generated from the current immediate tile; integer reduction and F32 order
//! are preserved. Each weight scale is loaded once per output tile/block.
//! Caller guarantees R<=64, C in {8,16}, valid padded q rows, and matching
//! weight/scales/output dimensions. Every weight load spans exactly8 bytes.
#[target_feature(enable="simd128")]
pub(super) unsafe fn accumulate<const R: usize, const C: usize>(
    q: *const i16,
    w: *const i8,
    cols: usize,
    start: usize,
    sx:*const f32, stride:usize, sw:*const f32, sums:&mut [[f32; C]; R],
) {
    assert!(R <= 64);
    const { assert!(C <= 16 && C % 4 == 0); }
    use core::arch::wasm32::*;
    let ws:[v128;4]=core::array::from_fn(|j| if j*4<C {v128_load(sw.add(j*4).cast())}else{f32x4_splat(0.)});
    let qp: [*const i16; R] = core::array::from_fn(|i| q.add(i * cols + start));
    let wp: [*const i8; C] = core::array::from_fn(|j| w.add(j * cols + start));
    for c in (0..256).step_by(256) {
        #[cfg(not(feature="experimental-explicit-weight-loads"))]
        let weights: [[v128; 32]; C] = core::array::from_fn(|j| {
            core::array::from_fn(|g| {
                i16x8_extend_low_i8x16(v128_load64_zero(wp[j].add(c + g * 8).cast()))
            })
        });
        #[cfg(feature="experimental-explicit-weight-loads")]
        let weights: [[v128;32];C] = {
            // All loads are within the caller's C checked weight rows and
            // their block256. Explicit columns prevent an outer from_fn loop
            // from materializing the entire expanded weight array in memory.
            let mut weights=[[i32x4_splat(0);32];C];
            macro_rules! column {($j:literal)=>{if C>$j {weights[$j]=[
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 0).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 8).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 16).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 24).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 32).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 40).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 48).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 56).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 64).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 72).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 80).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 88).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 96).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 104).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 112).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 120).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 128).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 136).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 144).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 152).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 160).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 168).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 176).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 184).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 192).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 200).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 208).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 216).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 224).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 232).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 240).cast())),
                i16x8_extend_low_i8x16(v128_load64_zero(wp[$j].add(c + 248).cast()))
            ];}};}
            column!(0);column!(1);column!(2);column!(3);
            column!(4);column!(5);column!(6);column!(7);
            column!(8);column!(9);column!(10);column!(11);
            column!(12);column!(13);column!(14);column!(15);
            weights
        };
        macro_rules! dot {
            ($input:ident,$j:expr) => {
                i32x4_add(
                    i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[0], weights[$j][0]),
                                    i32x4_dot_i16x8($input[1], weights[$j][1]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[2], weights[$j][2]),
                                    i32x4_dot_i16x8($input[3], weights[$j][3]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[4], weights[$j][4]),
                                    i32x4_dot_i16x8($input[5], weights[$j][5]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[6], weights[$j][6]),
                                    i32x4_dot_i16x8($input[7], weights[$j][7]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[8], weights[$j][8]),
                                    i32x4_dot_i16x8($input[9], weights[$j][9]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[10], weights[$j][10]),
                                    i32x4_dot_i16x8($input[11], weights[$j][11]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[12], weights[$j][12]),
                                    i32x4_dot_i16x8($input[13], weights[$j][13]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[14], weights[$j][14]),
                                    i32x4_dot_i16x8($input[15], weights[$j][15]),
                                ),
                            ),
                        ),
                    ),
                    i32x4_add(
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[16], weights[$j][16]),
                                    i32x4_dot_i16x8($input[17], weights[$j][17]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[18], weights[$j][18]),
                                    i32x4_dot_i16x8($input[19], weights[$j][19]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[20], weights[$j][20]),
                                    i32x4_dot_i16x8($input[21], weights[$j][21]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[22], weights[$j][22]),
                                    i32x4_dot_i16x8($input[23], weights[$j][23]),
                                ),
                            ),
                        ),
                        i32x4_add(
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[24], weights[$j][24]),
                                    i32x4_dot_i16x8($input[25], weights[$j][25]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[26], weights[$j][26]),
                                    i32x4_dot_i16x8($input[27], weights[$j][27]),
                                ),
                            ),
                            i32x4_add(
                                i32x4_add(
                                    i32x4_dot_i16x8($input[28], weights[$j][28]),
                                    i32x4_dot_i16x8($input[29], weights[$j][29]),
                                ),
                                i32x4_add(
                                    i32x4_dot_i16x8($input[30], weights[$j][30]),
                                    i32x4_dot_i16x8($input[31], weights[$j][31]),
                                ),
                            ),
                        ),
                    ),
                )
            };
        }
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
                    let scaled=f32x4_mul(f32x4_mul(f32x4_convert_i32x4(total),f32x4_splat(*sx.add($i*stride))),ws[$j/4]);
                    let dst=sums[$i].as_mut_ptr().add($j);
                    v128_store(dst.cast(),f32x4_add(v128_load(dst.cast()),scaled));
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
}

