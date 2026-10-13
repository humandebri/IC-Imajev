//! Diagnostic scalar oracle, independent of the production GQA/SIMD helpers.
use std::cell::Cell;
thread_local! {static ENABLED:Cell<bool>=const {Cell::new(false)};}
pub fn enabled()->bool {ENABLED.with(Cell::get)}
pub fn set(enabled:bool) {ENABLED.with(|v|v.set(enabled));}
pub(crate) fn head(q:&[f32],k:&[f32],v:&[f32],n:usize,width:usize,prefix:usize)->Vec<f32> {
    assert_eq!(q.len(),n*width);assert_eq!(k.len(),(n+prefix)*width);assert_eq!(v.len(),k.len());
    let mut out=vec![0.;n*width];let scale=(width as f32).sqrt();
    for token in 0..n {
        let mut scores=Vec::with_capacity(prefix+token+1);
        for previous in 0..=prefix+token {
            let mut dot=-0.0f32;
            for column in 0..width {dot+=q[token*width+column]*k[previous*width+column];}
            scores.push(crate::bf(dot/scale));
        }
        let maximum=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        for score in &mut scores {*score=(*score-maximum).exp();}
        let denominator=scores.iter().sum::<f32>();
        for column in 0..width {
            let mut sum=-0.0f32;
            for previous in 0..scores.len() {sum+=crate::bf(scores[previous]/denominator)*v[previous*width+column];}
            out[token*width+column]=crate::bf(sum);
        }
    }
    out
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn long_history_views_match_independent_scalar_reference() {
        for (n,total,width) in [(1,1024,256),(2,1024,256),(57,1024,256),(57,823,256),(7,1024,255)] {
            let q:Vec<_>=(0..n*width).map(|i|crate::bf((i%17) as f32/128.-0.0625)).collect();
            let k:Vec<_>=(0..total*width).map(|i|crate::bf((i%23) as f32/128.-0.09375)).collect();
            let v:Vec<_>=(0..total*width).map(|i|crate::bf((i%29) as f32/128.-0.109375)).collect();
            let want=head(&q,&k,&v,n,width,total-n);
            let got=crate::attention_views::head(&q,&k,&v,n,width,total-n);
            assert_eq!(got.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),want.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        }
    }
}
