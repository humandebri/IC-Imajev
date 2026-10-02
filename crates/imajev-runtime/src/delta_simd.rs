//! Four independent value lanes retain the scalar per-key sum order.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn run(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    use core::arch::wasm32::*;
    let mut transposed = vec![0f32; dk * dv];
    for i in 0..dk {
        for d in 0..dv {
            transposed[i * dv + d] = state[d * dk + i];
        }
    }
    let sp = transposed.as_mut_ptr();
    let mut out = vec![0.; g.len() * dv];
    for t in 0..g.len() {
        let decay = f32x4_splat(g[t]);
        let b = f32x4_splat(beta[t]);
        for d in (0..dv).step_by(4) {
            let mut mem = f32x4_splat(0.);
            for i in 0..dk {
                let p = sp.add(i * dv + d);
                let s = f32x4_mul(v128_load(p.cast()), decay);
                v128_store(p.cast(), s);
                mem = f32x4_add(mem, f32x4_mul(s, f32x4_splat(*k.get_unchecked(t * dk + i))));
            }
            let update = f32x4_mul(
                f32x4_sub(v128_load(v.as_ptr().add(t * dv + d).cast()), mem),
                b,
            );
            let mut o = f32x4_splat(0.);
            for i in 0..dk {
                let p = sp.add(i * dv + d);
                let s = f32x4_add(
                    v128_load(p.cast()),
                    f32x4_mul(f32x4_splat(*k.get_unchecked(t * dk + i)), update),
                );
                v128_store(p.cast(), s);
                o = f32x4_add(o, f32x4_mul(s, f32x4_splat(*q.get_unchecked(t * dk + i))));
            }
            v128_store(out.as_mut_ptr().add(t * dv + d).cast(), o);
        }
    }
    for i in 0..dk {
        for d in 0..dv {
            state[d * dk + i] = transposed[i * dv + d];
        }
    }
    out
}
