//! Record the original F32 innovation while computing the original output.
//! Derived from delta_simd::run; fixed 128x128 shape, no repeated recurrence.
//! Four independent value lanes retain the scalar per-key sum order.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn run_recorded(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    dk: usize,
    dv: usize,
) -> (Vec<f32>,Vec<f32>) {
    assert_eq!((dk,dv),(128,128));
    use core::arch::wasm32::*;
    let mut transposed = vec![0f32; dk * dv];
    for i in 0..dk {
        for d in 0..dv {
            transposed[i * dv + d] = state[d * dk + i];
        }
    }
    let sp = transposed.as_mut_ptr();
    let mut out = vec![0.; g.len() * dv];
    let mut innovations = vec![0.; g.len() * dv];
    for t in 0..g.len() {
        let decay = f32x4_splat(g[t]);
        let b = f32x4_splat(beta[t]);
        for start in (0..dv).step_by(128) {
            if start + 128 <= dv {
                let mut mem0 = f32x4_splat(0.);
                let mut mem1 = f32x4_splat(0.);
                let mut mem2 = f32x4_splat(0.);
                let mut mem3 = f32x4_splat(0.);
                let mut mem4 = f32x4_splat(0.);
                let mut mem5 = f32x4_splat(0.);
                let mut mem6 = f32x4_splat(0.);
                let mut mem7 = f32x4_splat(0.);
                let mut mem8 = f32x4_splat(0.);
                let mut mem9 = f32x4_splat(0.);
                let mut mem10 = f32x4_splat(0.);
                let mut mem11 = f32x4_splat(0.);
                let mut mem12 = f32x4_splat(0.);
                let mut mem13 = f32x4_splat(0.);
                let mut mem14 = f32x4_splat(0.);
                let mut mem15 = f32x4_splat(0.);
                let mut mem16 = f32x4_splat(0.);
                let mut mem17 = f32x4_splat(0.);
                let mut mem18 = f32x4_splat(0.);
                let mut mem19 = f32x4_splat(0.);
                let mut mem20 = f32x4_splat(0.);
                let mut mem21 = f32x4_splat(0.);
                let mut mem22 = f32x4_splat(0.);
                let mut mem23 = f32x4_splat(0.);
                let mut mem24 = f32x4_splat(0.);
                let mut mem25 = f32x4_splat(0.);
                let mut mem26 = f32x4_splat(0.);
                let mut mem27 = f32x4_splat(0.);
                let mut mem28 = f32x4_splat(0.);
                let mut mem29 = f32x4_splat(0.);
                let mut mem30 = f32x4_splat(0.);
                let mut mem31 = f32x4_splat(0.);
                for i in 0..dk {
                    let p = sp.add(i * dv + start);
                    let ki = f32x4_splat(*k.get_unchecked(t * dk + i));
                    mem0 = f32x4_add(
                        mem0,
                        f32x4_mul(f32x4_mul(v128_load(p.add(0).cast()), decay), ki),
                    );
                    mem1 = f32x4_add(
                        mem1,
                        f32x4_mul(f32x4_mul(v128_load(p.add(4).cast()), decay), ki),
                    );
                    mem2 = f32x4_add(
                        mem2,
                        f32x4_mul(f32x4_mul(v128_load(p.add(8).cast()), decay), ki),
                    );
                    mem3 = f32x4_add(
                        mem3,
                        f32x4_mul(f32x4_mul(v128_load(p.add(12).cast()), decay), ki),
                    );
                    mem4 = f32x4_add(
                        mem4,
                        f32x4_mul(f32x4_mul(v128_load(p.add(16).cast()), decay), ki),
                    );
                    mem5 = f32x4_add(
                        mem5,
                        f32x4_mul(f32x4_mul(v128_load(p.add(20).cast()), decay), ki),
                    );
                    mem6 = f32x4_add(
                        mem6,
                        f32x4_mul(f32x4_mul(v128_load(p.add(24).cast()), decay), ki),
                    );
                    mem7 = f32x4_add(
                        mem7,
                        f32x4_mul(f32x4_mul(v128_load(p.add(28).cast()), decay), ki),
                    );
                    mem8 = f32x4_add(
                        mem8,
                        f32x4_mul(f32x4_mul(v128_load(p.add(32).cast()), decay), ki),
                    );
                    mem9 = f32x4_add(
                        mem9,
                        f32x4_mul(f32x4_mul(v128_load(p.add(36).cast()), decay), ki),
                    );
                    mem10 = f32x4_add(
                        mem10,
                        f32x4_mul(f32x4_mul(v128_load(p.add(40).cast()), decay), ki),
                    );
                    mem11 = f32x4_add(
                        mem11,
                        f32x4_mul(f32x4_mul(v128_load(p.add(44).cast()), decay), ki),
                    );
                    mem12 = f32x4_add(
                        mem12,
                        f32x4_mul(f32x4_mul(v128_load(p.add(48).cast()), decay), ki),
                    );
                    mem13 = f32x4_add(
                        mem13,
                        f32x4_mul(f32x4_mul(v128_load(p.add(52).cast()), decay), ki),
                    );
                    mem14 = f32x4_add(
                        mem14,
                        f32x4_mul(f32x4_mul(v128_load(p.add(56).cast()), decay), ki),
                    );
                    mem15 = f32x4_add(
                        mem15,
                        f32x4_mul(f32x4_mul(v128_load(p.add(60).cast()), decay), ki),
                    );
                    mem16 = f32x4_add(
                        mem16,
                        f32x4_mul(f32x4_mul(v128_load(p.add(64).cast()), decay), ki),
                    );
                    mem17 = f32x4_add(
                        mem17,
                        f32x4_mul(f32x4_mul(v128_load(p.add(68).cast()), decay), ki),
                    );
                    mem18 = f32x4_add(
                        mem18,
                        f32x4_mul(f32x4_mul(v128_load(p.add(72).cast()), decay), ki),
                    );
                    mem19 = f32x4_add(
                        mem19,
                        f32x4_mul(f32x4_mul(v128_load(p.add(76).cast()), decay), ki),
                    );
                    mem20 = f32x4_add(
                        mem20,
                        f32x4_mul(f32x4_mul(v128_load(p.add(80).cast()), decay), ki),
                    );
                    mem21 = f32x4_add(
                        mem21,
                        f32x4_mul(f32x4_mul(v128_load(p.add(84).cast()), decay), ki),
                    );
                    mem22 = f32x4_add(
                        mem22,
                        f32x4_mul(f32x4_mul(v128_load(p.add(88).cast()), decay), ki),
                    );
                    mem23 = f32x4_add(
                        mem23,
                        f32x4_mul(f32x4_mul(v128_load(p.add(92).cast()), decay), ki),
                    );
                    mem24 = f32x4_add(
                        mem24,
                        f32x4_mul(f32x4_mul(v128_load(p.add(96).cast()), decay), ki),
                    );
                    mem25 = f32x4_add(
                        mem25,
                        f32x4_mul(f32x4_mul(v128_load(p.add(100).cast()), decay), ki),
                    );
                    mem26 = f32x4_add(
                        mem26,
                        f32x4_mul(f32x4_mul(v128_load(p.add(104).cast()), decay), ki),
                    );
                    mem27 = f32x4_add(
                        mem27,
                        f32x4_mul(f32x4_mul(v128_load(p.add(108).cast()), decay), ki),
                    );
                    mem28 = f32x4_add(
                        mem28,
                        f32x4_mul(f32x4_mul(v128_load(p.add(112).cast()), decay), ki),
                    );
                    mem29 = f32x4_add(
                        mem29,
                        f32x4_mul(f32x4_mul(v128_load(p.add(116).cast()), decay), ki),
                    );
                    mem30 = f32x4_add(
                        mem30,
                        f32x4_mul(f32x4_mul(v128_load(p.add(120).cast()), decay), ki),
                    );
                    mem31 = f32x4_add(
                        mem31,
                        f32x4_mul(f32x4_mul(v128_load(p.add(124).cast()), decay), ki),
                    );
                }
                let update0 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 0).cast()), mem0),
                    b,
                );
                let mut o0 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 0).cast(), update0);
                let update1 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 4).cast()), mem1),
                    b,
                );
                let mut o1 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 4).cast(), update1);
                let update2 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 8).cast()), mem2),
                    b,
                );
                let mut o2 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 8).cast(), update2);
                let update3 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 12).cast()), mem3),
                    b,
                );
                let mut o3 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 12).cast(), update3);
                let update4 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 16).cast()), mem4),
                    b,
                );
                let mut o4 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 16).cast(), update4);
                let update5 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 20).cast()), mem5),
                    b,
                );
                let mut o5 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 20).cast(), update5);
                let update6 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 24).cast()), mem6),
                    b,
                );
                let mut o6 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 24).cast(), update6);
                let update7 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 28).cast()), mem7),
                    b,
                );
                let mut o7 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 28).cast(), update7);
                let update8 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 32).cast()), mem8),
                    b,
                );
                let mut o8 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 32).cast(), update8);
                let update9 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 36).cast()), mem9),
                    b,
                );
                let mut o9 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 36).cast(), update9);
                let update10 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 40).cast()), mem10),
                    b,
                );
                let mut o10 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 40).cast(), update10);
                let update11 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 44).cast()), mem11),
                    b,
                );
                let mut o11 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 44).cast(), update11);
                let update12 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 48).cast()), mem12),
                    b,
                );
                let mut o12 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 48).cast(), update12);
                let update13 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 52).cast()), mem13),
                    b,
                );
                let mut o13 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 52).cast(), update13);
                let update14 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 56).cast()), mem14),
                    b,
                );
                let mut o14 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 56).cast(), update14);
                let update15 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 60).cast()), mem15),
                    b,
                );
                let mut o15 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 60).cast(), update15);
                let update16 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 64).cast()), mem16),
                    b,
                );
                let mut o16 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 64).cast(), update16);
                let update17 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 68).cast()), mem17),
                    b,
                );
                let mut o17 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 68).cast(), update17);
                let update18 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 72).cast()), mem18),
                    b,
                );
                let mut o18 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 72).cast(), update18);
                let update19 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 76).cast()), mem19),
                    b,
                );
                let mut o19 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 76).cast(), update19);
                let update20 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 80).cast()), mem20),
                    b,
                );
                let mut o20 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 80).cast(), update20);
                let update21 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 84).cast()), mem21),
                    b,
                );
                let mut o21 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 84).cast(), update21);
                let update22 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 88).cast()), mem22),
                    b,
                );
                let mut o22 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 88).cast(), update22);
                let update23 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 92).cast()), mem23),
                    b,
                );
                let mut o23 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 92).cast(), update23);
                let update24 = f32x4_mul(
                    f32x4_sub(v128_load(v.as_ptr().add(t * dv + start + 96).cast()), mem24),
                    b,
                );
                let mut o24 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 96).cast(), update24);
                let update25 = f32x4_mul(
                    f32x4_sub(
                        v128_load(v.as_ptr().add(t * dv + start + 100).cast()),
                        mem25,
                    ),
                    b,
                );
                let mut o25 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 100).cast(), update25);
                let update26 = f32x4_mul(
                    f32x4_sub(
                        v128_load(v.as_ptr().add(t * dv + start + 104).cast()),
                        mem26,
                    ),
                    b,
                );
                let mut o26 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 104).cast(), update26);
                let update27 = f32x4_mul(
                    f32x4_sub(
                        v128_load(v.as_ptr().add(t * dv + start + 108).cast()),
                        mem27,
                    ),
                    b,
                );
                let mut o27 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 108).cast(), update27);
                let update28 = f32x4_mul(
                    f32x4_sub(
                        v128_load(v.as_ptr().add(t * dv + start + 112).cast()),
                        mem28,
                    ),
                    b,
                );
                let mut o28 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 112).cast(), update28);
                let update29 = f32x4_mul(
                    f32x4_sub(
                        v128_load(v.as_ptr().add(t * dv + start + 116).cast()),
                        mem29,
                    ),
                    b,
                );
                let mut o29 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 116).cast(), update29);
                let update30 = f32x4_mul(
                    f32x4_sub(
                        v128_load(v.as_ptr().add(t * dv + start + 120).cast()),
                        mem30,
                    ),
                    b,
                );
                let mut o30 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 120).cast(), update30);
                let update31 = f32x4_mul(
                    f32x4_sub(
                        v128_load(v.as_ptr().add(t * dv + start + 124).cast()),
                        mem31,
                    ),
                    b,
                );
                let mut o31 = f32x4_splat(0.);
                v128_store(innovations.as_mut_ptr().add(t * dv + start + 124).cast(), update31);
                for i in 0..dk {
                    let p = sp.add(i * dv + start);
                    let ki = f32x4_splat(*k.get_unchecked(t * dk + i));
                    let qi = f32x4_splat(*q.get_unchecked(t * dk + i));
                    let s0 = f32x4_add(
                        f32x4_mul(v128_load(p.add(0).cast()), decay),
                        f32x4_mul(ki, update0),
                    );
                    v128_store(p.add(0).cast(), s0);
                    o0 = f32x4_add(o0, f32x4_mul(s0, qi));
                    let s1 = f32x4_add(
                        f32x4_mul(v128_load(p.add(4).cast()), decay),
                        f32x4_mul(ki, update1),
                    );
                    v128_store(p.add(4).cast(), s1);
                    o1 = f32x4_add(o1, f32x4_mul(s1, qi));
                    let s2 = f32x4_add(
                        f32x4_mul(v128_load(p.add(8).cast()), decay),
                        f32x4_mul(ki, update2),
                    );
                    v128_store(p.add(8).cast(), s2);
                    o2 = f32x4_add(o2, f32x4_mul(s2, qi));
                    let s3 = f32x4_add(
                        f32x4_mul(v128_load(p.add(12).cast()), decay),
                        f32x4_mul(ki, update3),
                    );
                    v128_store(p.add(12).cast(), s3);
                    o3 = f32x4_add(o3, f32x4_mul(s3, qi));
                    let s4 = f32x4_add(
                        f32x4_mul(v128_load(p.add(16).cast()), decay),
                        f32x4_mul(ki, update4),
                    );
                    v128_store(p.add(16).cast(), s4);
                    o4 = f32x4_add(o4, f32x4_mul(s4, qi));
                    let s5 = f32x4_add(
                        f32x4_mul(v128_load(p.add(20).cast()), decay),
                        f32x4_mul(ki, update5),
                    );
                    v128_store(p.add(20).cast(), s5);
                    o5 = f32x4_add(o5, f32x4_mul(s5, qi));
                    let s6 = f32x4_add(
                        f32x4_mul(v128_load(p.add(24).cast()), decay),
                        f32x4_mul(ki, update6),
                    );
                    v128_store(p.add(24).cast(), s6);
                    o6 = f32x4_add(o6, f32x4_mul(s6, qi));
                    let s7 = f32x4_add(
                        f32x4_mul(v128_load(p.add(28).cast()), decay),
                        f32x4_mul(ki, update7),
                    );
                    v128_store(p.add(28).cast(), s7);
                    o7 = f32x4_add(o7, f32x4_mul(s7, qi));
                    let s8 = f32x4_add(
                        f32x4_mul(v128_load(p.add(32).cast()), decay),
                        f32x4_mul(ki, update8),
                    );
                    v128_store(p.add(32).cast(), s8);
                    o8 = f32x4_add(o8, f32x4_mul(s8, qi));
                    let s9 = f32x4_add(
                        f32x4_mul(v128_load(p.add(36).cast()), decay),
                        f32x4_mul(ki, update9),
                    );
                    v128_store(p.add(36).cast(), s9);
                    o9 = f32x4_add(o9, f32x4_mul(s9, qi));
                    let s10 = f32x4_add(
                        f32x4_mul(v128_load(p.add(40).cast()), decay),
                        f32x4_mul(ki, update10),
                    );
                    v128_store(p.add(40).cast(), s10);
                    o10 = f32x4_add(o10, f32x4_mul(s10, qi));
                    let s11 = f32x4_add(
                        f32x4_mul(v128_load(p.add(44).cast()), decay),
                        f32x4_mul(ki, update11),
                    );
                    v128_store(p.add(44).cast(), s11);
                    o11 = f32x4_add(o11, f32x4_mul(s11, qi));
                    let s12 = f32x4_add(
                        f32x4_mul(v128_load(p.add(48).cast()), decay),
                        f32x4_mul(ki, update12),
                    );
                    v128_store(p.add(48).cast(), s12);
                    o12 = f32x4_add(o12, f32x4_mul(s12, qi));
                    let s13 = f32x4_add(
                        f32x4_mul(v128_load(p.add(52).cast()), decay),
                        f32x4_mul(ki, update13),
                    );
                    v128_store(p.add(52).cast(), s13);
                    o13 = f32x4_add(o13, f32x4_mul(s13, qi));
                    let s14 = f32x4_add(
                        f32x4_mul(v128_load(p.add(56).cast()), decay),
                        f32x4_mul(ki, update14),
                    );
                    v128_store(p.add(56).cast(), s14);
                    o14 = f32x4_add(o14, f32x4_mul(s14, qi));
                    let s15 = f32x4_add(
                        f32x4_mul(v128_load(p.add(60).cast()), decay),
                        f32x4_mul(ki, update15),
                    );
                    v128_store(p.add(60).cast(), s15);
                    o15 = f32x4_add(o15, f32x4_mul(s15, qi));
                    let s16 = f32x4_add(
                        f32x4_mul(v128_load(p.add(64).cast()), decay),
                        f32x4_mul(ki, update16),
                    );
                    v128_store(p.add(64).cast(), s16);
                    o16 = f32x4_add(o16, f32x4_mul(s16, qi));
                    let s17 = f32x4_add(
                        f32x4_mul(v128_load(p.add(68).cast()), decay),
                        f32x4_mul(ki, update17),
                    );
                    v128_store(p.add(68).cast(), s17);
                    o17 = f32x4_add(o17, f32x4_mul(s17, qi));
                    let s18 = f32x4_add(
                        f32x4_mul(v128_load(p.add(72).cast()), decay),
                        f32x4_mul(ki, update18),
                    );
                    v128_store(p.add(72).cast(), s18);
                    o18 = f32x4_add(o18, f32x4_mul(s18, qi));
                    let s19 = f32x4_add(
                        f32x4_mul(v128_load(p.add(76).cast()), decay),
                        f32x4_mul(ki, update19),
                    );
                    v128_store(p.add(76).cast(), s19);
                    o19 = f32x4_add(o19, f32x4_mul(s19, qi));
                    let s20 = f32x4_add(
                        f32x4_mul(v128_load(p.add(80).cast()), decay),
                        f32x4_mul(ki, update20),
                    );
                    v128_store(p.add(80).cast(), s20);
                    o20 = f32x4_add(o20, f32x4_mul(s20, qi));
                    let s21 = f32x4_add(
                        f32x4_mul(v128_load(p.add(84).cast()), decay),
                        f32x4_mul(ki, update21),
                    );
                    v128_store(p.add(84).cast(), s21);
                    o21 = f32x4_add(o21, f32x4_mul(s21, qi));
                    let s22 = f32x4_add(
                        f32x4_mul(v128_load(p.add(88).cast()), decay),
                        f32x4_mul(ki, update22),
                    );
                    v128_store(p.add(88).cast(), s22);
                    o22 = f32x4_add(o22, f32x4_mul(s22, qi));
                    let s23 = f32x4_add(
                        f32x4_mul(v128_load(p.add(92).cast()), decay),
                        f32x4_mul(ki, update23),
                    );
                    v128_store(p.add(92).cast(), s23);
                    o23 = f32x4_add(o23, f32x4_mul(s23, qi));
                    let s24 = f32x4_add(
                        f32x4_mul(v128_load(p.add(96).cast()), decay),
                        f32x4_mul(ki, update24),
                    );
                    v128_store(p.add(96).cast(), s24);
                    o24 = f32x4_add(o24, f32x4_mul(s24, qi));
                    let s25 = f32x4_add(
                        f32x4_mul(v128_load(p.add(100).cast()), decay),
                        f32x4_mul(ki, update25),
                    );
                    v128_store(p.add(100).cast(), s25);
                    o25 = f32x4_add(o25, f32x4_mul(s25, qi));
                    let s26 = f32x4_add(
                        f32x4_mul(v128_load(p.add(104).cast()), decay),
                        f32x4_mul(ki, update26),
                    );
                    v128_store(p.add(104).cast(), s26);
                    o26 = f32x4_add(o26, f32x4_mul(s26, qi));
                    let s27 = f32x4_add(
                        f32x4_mul(v128_load(p.add(108).cast()), decay),
                        f32x4_mul(ki, update27),
                    );
                    v128_store(p.add(108).cast(), s27);
                    o27 = f32x4_add(o27, f32x4_mul(s27, qi));
                    let s28 = f32x4_add(
                        f32x4_mul(v128_load(p.add(112).cast()), decay),
                        f32x4_mul(ki, update28),
                    );
                    v128_store(p.add(112).cast(), s28);
                    o28 = f32x4_add(o28, f32x4_mul(s28, qi));
                    let s29 = f32x4_add(
                        f32x4_mul(v128_load(p.add(116).cast()), decay),
                        f32x4_mul(ki, update29),
                    );
                    v128_store(p.add(116).cast(), s29);
                    o29 = f32x4_add(o29, f32x4_mul(s29, qi));
                    let s30 = f32x4_add(
                        f32x4_mul(v128_load(p.add(120).cast()), decay),
                        f32x4_mul(ki, update30),
                    );
                    v128_store(p.add(120).cast(), s30);
                    o30 = f32x4_add(o30, f32x4_mul(s30, qi));
                    let s31 = f32x4_add(
                        f32x4_mul(v128_load(p.add(124).cast()), decay),
                        f32x4_mul(ki, update31),
                    );
                    v128_store(p.add(124).cast(), s31);
                    o31 = f32x4_add(o31, f32x4_mul(s31, qi));
                }
                v128_store(out.as_mut_ptr().add(t * dv + start + 0).cast(), o0);
                v128_store(out.as_mut_ptr().add(t * dv + start + 4).cast(), o1);
                v128_store(out.as_mut_ptr().add(t * dv + start + 8).cast(), o2);
                v128_store(out.as_mut_ptr().add(t * dv + start + 12).cast(), o3);
                v128_store(out.as_mut_ptr().add(t * dv + start + 16).cast(), o4);
                v128_store(out.as_mut_ptr().add(t * dv + start + 20).cast(), o5);
                v128_store(out.as_mut_ptr().add(t * dv + start + 24).cast(), o6);
                v128_store(out.as_mut_ptr().add(t * dv + start + 28).cast(), o7);
                v128_store(out.as_mut_ptr().add(t * dv + start + 32).cast(), o8);
                v128_store(out.as_mut_ptr().add(t * dv + start + 36).cast(), o9);
                v128_store(out.as_mut_ptr().add(t * dv + start + 40).cast(), o10);
                v128_store(out.as_mut_ptr().add(t * dv + start + 44).cast(), o11);
                v128_store(out.as_mut_ptr().add(t * dv + start + 48).cast(), o12);
                v128_store(out.as_mut_ptr().add(t * dv + start + 52).cast(), o13);
                v128_store(out.as_mut_ptr().add(t * dv + start + 56).cast(), o14);
                v128_store(out.as_mut_ptr().add(t * dv + start + 60).cast(), o15);
                v128_store(out.as_mut_ptr().add(t * dv + start + 64).cast(), o16);
                v128_store(out.as_mut_ptr().add(t * dv + start + 68).cast(), o17);
                v128_store(out.as_mut_ptr().add(t * dv + start + 72).cast(), o18);
                v128_store(out.as_mut_ptr().add(t * dv + start + 76).cast(), o19);
                v128_store(out.as_mut_ptr().add(t * dv + start + 80).cast(), o20);
                v128_store(out.as_mut_ptr().add(t * dv + start + 84).cast(), o21);
                v128_store(out.as_mut_ptr().add(t * dv + start + 88).cast(), o22);
                v128_store(out.as_mut_ptr().add(t * dv + start + 92).cast(), o23);
                v128_store(out.as_mut_ptr().add(t * dv + start + 96).cast(), o24);
                v128_store(out.as_mut_ptr().add(t * dv + start + 100).cast(), o25);
                v128_store(out.as_mut_ptr().add(t * dv + start + 104).cast(), o26);
                v128_store(out.as_mut_ptr().add(t * dv + start + 108).cast(), o27);
                v128_store(out.as_mut_ptr().add(t * dv + start + 112).cast(), o28);
                v128_store(out.as_mut_ptr().add(t * dv + start + 116).cast(), o29);
                v128_store(out.as_mut_ptr().add(t * dv + start + 120).cast(), o30);
                v128_store(out.as_mut_ptr().add(t * dv + start + 124).cast(), o31);
            } else {
                for d in (start..dv).step_by(4) {
                    let mut mem = f32x4_splat(0.);
                    for i in 0..dk {
                        let p = sp.add(i * dv + d);
                        let s = f32x4_mul(v128_load(p.cast()), decay);
                        mem =
                            f32x4_add(mem, f32x4_mul(s, f32x4_splat(*k.get_unchecked(t * dk + i))));
                    }
                    let update = f32x4_mul(
                        f32x4_sub(v128_load(v.as_ptr().add(t * dv + d).cast()), mem),
                        b,
                    );
                    let mut o = f32x4_splat(0.);
                    for i in 0..dk {
                        let p = sp.add(i * dv + d);
                        let s = f32x4_add(
                            // Recompute the same rounded decay rather than writing and
                            // reloading an intermediate state for every token.
                            f32x4_mul(v128_load(p.cast()), decay),
                            f32x4_mul(f32x4_splat(*k.get_unchecked(t * dk + i)), update),
                        );
                        v128_store(p.cast(), s);
                        o = f32x4_add(o, f32x4_mul(s, f32x4_splat(*q.get_unchecked(t * dk + i))));
                    }
                    v128_store(out.as_mut_ptr().add(t * dv + d).cast(), o);
                }
            }
        }
    }
    for i in 0..dk {
        for d in 0..dv {
            state[d * dk + i] = transposed[i * dv + d];
        }
    }
    (out,innovations)
}
