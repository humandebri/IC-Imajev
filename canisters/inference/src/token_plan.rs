//! Bounded operator schedules. Short inputs keep the existing bulk graph.
pub(super) const ATTENTION_TOKENS:usize=57;
#[cfg(feature="experimental-adaptive-token-tiles")]
const DENSE_TOKENS:usize=89;
#[cfg(not(feature="experimental-adaptive-token-tiles"))]
const DENSE_TOKENS:usize=57;

#[derive(Clone,Copy,Debug)]
pub(super) struct Tile {
    pub layer:usize, pub mlp:bool, pub begin:usize, pub count:usize,
    pub first:bool, pub last:bool,
}
pub(super) fn stages(n:usize)->u64 {
    if n<=89 {return 64;}
    (24*n.div_ceil(DENSE_TOKENS)+8*n.div_ceil(ATTENTION_TOKENS)+31*n.div_ceil(DENSE_TOKENS)+1) as u64
}
pub(super) fn schedule(n:usize)->Vec<Tile> {
    let mut out=Vec::with_capacity(stages(n) as usize);
    for layer in 0..32 {
        for mlp in [false,true] {
            let tokens=if mlp && layer==31 {1}else{n};
            let cap=if !mlp && layer%4==3 {ATTENTION_TOKENS}else{DENSE_TOKENS};
            for begin in (0..tokens).step_by(cap) {
                let count=(tokens-begin).min(cap);
                out.push(Tile{layer,mlp,begin,count,first:begin==0,last:begin+count==tokens});
            }
        }
    }
    out
}

#[cfg(test)]mod tests {
    use super::*;
    #[test]fn all_supported_lengths_cover_each_operator_exactly_once() {
        for n in 90..=485 {
            let plan=schedule(n);
            assert_eq!(plan.len() as u64,stages(n));
            let mut at=0;
            for layer in 0..32 {for mlp in [false,true] {
                let total=if mlp && layer==31 {1}else{n};
                let mut covered=0;
                while at<plan.len() && plan[at].layer==layer && plan[at].mlp==mlp {
                    let t=plan[at];assert_eq!(t.begin,covered);assert_eq!(t.first,covered==0);
                    assert!((1..=if !mlp && layer%4==3 {57}else{DENSE_TOKENS}).contains(&t.count));
                    covered+=t.count;assert_eq!(t.last,covered==total);at+=1;
                }
                assert_eq!(covered,total);
            }}
            assert_eq!(at,plan.len());
        }
    }
    #[test]fn short_input_progress_is_unchanged() {
        for n in 1..=89 {assert_eq!(stages(n),64);}
    }
}
