//! Last attention and terminal MLP share a query; retain original BF16 boundaries.
use crate::{Manifest,Request,Result,prepared_weights::WeightBuffer};
const C:usize=2560;
const ROOT:&str="model.language_model.layers.31";
pub(crate) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    if r.op!="terminal_attention_mlp_integer" || !crate::lossless_encoding(&r.encoding)
        || r.dims.len()!=2 || !r.aux.is_empty() || !r.scalars.is_empty()
        || r.tensor!=format!("{ROOT}.self_attn.q_proj.weight") {
        return Err("terminal attention metadata".into());
    }
    let(n,p)=(r.dims[0],r.dims[1]);
    if !(1..=132).contains(&n) || p>512 || n+p>512
        || x.len()!=(n+1)*C+p*2048 || x.len()>crate::MAX_FLOATS
        || !x.iter().all(|v|v.is_finite()) {
        return Err("terminal attention input bounds".into());
    }
    evaluate_validated_parts(r,&x[..n*C],&x[n*C..(n+1)*C],&x[(n+1)*C..],m,read)
}
// Internal callers already establish metadata, segment lengths and finiteness:
// evaluate checks the complete input; terminal_tail combines checked decoder
// history with finite MLP outputs. These borrowed segments remain immutable.
pub(crate) fn evaluate_validated_parts<F,B>(r:&Request,normalized:&[f32],residual:&[f32],history:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    let(n,p)=(r.dims[0],r.dims[1]);
    debug_assert_eq!(normalized.len(),n*C);debug_assert_eq!(residual.len(),C);debug_assert_eq!(history.len(),p*2048);
    // Input norm and original last residual have distinct BF16 boundaries.
    // KV history uses the same head-major order as attention_full_integer.
    let mut ar=r.clone();ar.op="attention_full_integer".into();ar.dims=vec![n,p,1];
    let mut attention_input=Vec::with_capacity(n*C+p*2048);
    attention_input.extend_from_slice(normalized);
    attention_input.extend_from_slice(history);
    let(attention,ra)=crate::attention_full::evaluate(&ar,&attention_input,m,read)?;
    if attention.len()!=C+n*2048 {return Err("terminal attention result bounds".into());}
    let mut tr=r.clone();tr.op="terminal_mlp_integer".into();tr.dims=vec![1,C];
    tr.tensor=format!("{ROOT}.post_attention_layernorm.weight");
    let mut mlp_input=Vec::with_capacity(2*C);
    mlp_input.extend_from_slice(residual);
    mlp_input.extend_from_slice(&attention[..C]);
    let(mut output,rm)=crate::terminal::evaluate(&tr,&mlp_input,m,read)?;
    if output.len()!=2*C {return Err("terminal MLP result bounds".into());}
    output.extend_from_slice(&attention[C..]);
    Ok((output,ra+rm))
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn bad_metadata_and_ranges_fail_before_weight_reads() {
        let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
        let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,
            "pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,
            "op":"terminal_attention_mlp_integer","tensor":format!("{ROOT}.self_attn.q_proj.weight"),
            "dims":[1,0],"scalars":[],"encoding":"bf16-block256-exact-v1"})).unwrap();
        for dims in [vec![],vec![0,0],vec![133,0],vec![1,512],vec![usize::MAX,0]] {
            let mut bad=r.clone();bad.dims=dims;
            assert!(evaluate(&bad,&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());
        }
        for bad in ["model.language_model.layers.3.self_attn.q_proj.weight","model.language_model.layers.031.self_attn.q_proj.weight"] {
            let mut changed=r.clone();changed.tensor=bad.into();
            assert!(evaluate(&changed,&vec![0.;2*C],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());
        }
        let mut bad=r.clone();bad.aux.push("unexpected".into());
        assert!(evaluate(&bad,&vec![0.;2*C],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());
        let mut x=vec![0.;2*C];x[C]=f32::NAN;
        assert!(evaluate(&r,&x,&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());
    }
}
