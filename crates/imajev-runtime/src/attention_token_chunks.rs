//! Server-only Attention tiles with bounded KV groups and shared Q preparation.
use crate::{prepared_weights::WeightBuffer, Manifest, Request, Result, MAX_FLOATS};

const C: usize = 2560;
const PREFIX: usize = 27;
const HEADS: usize = 8;

/// Preserve the ordinary full-head path whenever both its buffers and work fit.
pub fn needs_split(n: usize, offset: usize, last: bool) -> bool {
    let qn = if last { 1 } else { n };
    n * C + offset * 2048 > MAX_FLOATS
        || qn * C + (offset + n) * 2048 > MAX_FLOATS
        || qn * 16 * 256 + (offset + n) * 2048 > MAX_FLOATS
        || 16 * (qn * (qn + 1) / 2 + qn * (offset + n - qn)) * 256 > 75_000_000
}

// Prefix is head-major; previous and current suffix KV are token-major.
fn group_history(prefix: &[f32], keys: &[f32], values: &[f32], current: &[f32],
                 n: usize, total: usize, first: usize) -> Vec<f32> {
    let seen = keys.len() / 1024;
    let mut out = Vec::with_capacity(2 * (HEADS / 4) * total * 256);
    for part in 0..2 {
        let previous = if part == 0 { keys } else { values };
        for head in first / 4..(first + HEADS) / 4 {
            let at = part * PREFIX * 1024 + head * PREFIX * 256;
            out.extend_from_slice(&prefix[at..at + PREFIX * 256]);
            for token in 0..seen {
                let at = token * 1024 + head * 256;
                out.extend_from_slice(&previous[at..at + 256]);
            }
            for token in 0..n {
                let at = part * n * 1024 + token * 1024 + head * 256;
                out.extend_from_slice(&current[at..at + 256]);
            }
        }
    }
    out
}

pub fn evaluate<F, B>(r: &Request, norm: &[f32], prefix: &[f32], keys: &[f32],
                      values: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where F: FnMut(u64, usize) -> Result<B>, B: WeightBuffer {
    if r.op != "attention_full_integer" || !crate::lossless_encoding(&r.encoding)
        || r.dims.len() != 3 || !r.aux.is_empty() || !r.scalars.is_empty()
        || r.model != m.model || r.pack_hash != m.pack_hash {
        return Err("server Attention metadata/scope".into());
    }
    let (n, offset, last) = (r.dims[0], r.dims[1], r.dims[2]);
    if !(1..=57).contains(&n) || !(PREFIX..=512).contains(&offset)
        || offset + n > 512 || last > 1 || norm.len() != n * C
        || prefix.len() != PREFIX * 2048 || keys.len() != (offset - PREFIX) * 1024
        || values.len() != keys.len()
        || [norm, prefix, keys, values].iter().any(|x| !x.iter().all(|v| v.is_finite())) {
        return Err("server Attention bounds/history".into());
    }
    let p = r.tensor.strip_suffix(".self_attn.q_proj.weight").ok_or("server Attention tensor")?;
    let i = p.strip_prefix("model.language_model.layers.").ok_or("server Attention layer")?;
    let layer: usize = i.parse().map_err(|_| "server Attention layer")?;
    if layer > 31 || layer % 4 != 3 || i != layer.to_string() || (last == 1 && layer != 31) {
        return Err("server Attention layer".into());
    }
    let root = format!("{p}.self_attn");
    // Check all fixed weights before reads, including projections used at the end.
    for (name, rows, cols, dtype) in [
        ("k_proj.weight",1024,C,"int8"),("v_proj.weight",1024,C,"int8"),
        ("q_proj.weight",8192,C,"int8"),("o_proj.weight",C,4096,"int8"),
        ("k_proj.lora_A.weight",64,C,"f32"),("v_proj.lora_A.weight",64,C,"f32"),
        ("q_proj.lora_A.weight",64,C,"f32"),("o_proj.lora_A.weight",64,4096,"f32"),
        ("k_proj.lora_B.weight",1024,64,"f32"),("v_proj.lora_B.weight",1024,64,"f32"),
        ("q_proj.lora_B.weight",8192,64,"f32"),("o_proj.lora_B.weight",C,64,"f32"),
        ("k_norm.weight",1,256,"bf16"),("q_norm.weight",1,256,"bf16")
    ] {
        let name = format!("{root}.{name}");
        let t = m.tensors.iter().find(|t| t.name == name).ok_or("server Attention missing weight")?;
        let bytes = if dtype == "f32" { rows * cols * 4 }
                    else if dtype == "bf16" { rows * cols * 2 } else { rows * cols + rows * 4 };
        if t.rows != rows || t.cols != cols || t.dtype != dtype || t.bytes != bytes as u64 {
            return Err("server Attention weight shape".into());
        }
    }
    let q = crate::int8_kernel::quantize_rows(norm, n, C)?;
    let mut kr = r.clone(); kr.op = "attention_kv_integer".into();
    kr.tensor = format!("{root}.k_proj.weight"); kr.dims = vec![n, offset];
    let (kv, mut bytes) = crate::attention_fusion::evaluate_shared(&kr, norm, m, read, Some(&q))?;
    if kv.len() != n * 2048 { return Err("server Attention KV output".into()); }
    let qn = if last == 1 { 1 } else { n };
    let qnorm = &norm[(n - qn) * C..];
    let terminal_q;
    let prepared = if qn == n { &q } else {
        terminal_q = crate::int8_kernel::quantize_rows(qnorm, qn, C)?; &terminal_q
    };
    let total = offset + n;
    let mut qr = r.clone(); qr.op = "attention_q_gqa_integer".into();
    qr.dims = vec![qn, total, total - qn, 0, HEADS];
    let (ax, used) = crate::attention_fusion::prepare_q_a(&qr, qnorm, m, read)?; bytes += used;
    let mut gated = vec![0.; qn * 4096];
    for first in [0, HEADS] {
        let payload = group_history(prefix, keys, values, &kv, n, total, first);
        qr.dims[3] = first;
        let (part, used) = crate::attention_fusion::evaluate_shared_with_a(
            &qr, &payload, m, read, Some(prepared), Some(&ax))?;
        bytes += used;
        if part.len() != qn * HEADS * 256 { return Err("server Attention group output".into()); }
        for token in 0..qn {
            let at = token * 4096 + first * 256;
            gated[at..at + HEADS * 256].copy_from_slice(&part[token * HEADS * 256..(token + 1) * HEADS * 256]);
        }
    }
    let mut out = r.clone(); out.op = "lora_integer".into();
    out.tensor = format!("{root}.o_proj.weight"); out.dims = vec![qn, C, 4096, 0];
    out.scalars = vec![2.];
    out.aux = vec![format!("{root}.o_proj.lora_A.weight"), format!("{root}.o_proj.lora_B.weight")];
    let (mut result, used) = crate::evaluate_integer_with_ax(&out, &gated, m, read, None, None)?;
    result.extend(kv);
    if result.len() != qn * C + n * 2048 { return Err("server Attention output".into()); }
    Ok((result, bytes + used))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_large_tiles_need_extra_groups() {
        for suffix in 90..=229 {
            for begin in (0..suffix).step_by(57) {
                for last in [false, true] { assert!(!needs_split((suffix-begin).min(57), PREFIX+begin, last)); }
            }
        }
        assert!(needs_split(57, 312, false)); // Work exceeds 75M before buffers fill.
        assert!(!needs_split(57, 312, true));
        assert!(needs_split(47, 312, false)); // Rotated Q + KV, not just original Q + KV.
        assert!(needs_split(29, 483, false));
        assert!(needs_split(29, 483, true)); // Even one Q needs a bounded KV group.
    }
    #[test]
    fn grouped_history_keeps_every_prefix_and_suffix_token_at_512() {
        let n=29; let seen=456; let total=512;
        let prefix:Vec<_>=(0..PREFIX*2048).map(|v|v as f32).collect();
        let keys:Vec<_>=(0..seen*1024).map(|v|1_000_000.+v as f32).collect();
        let values:Vec<_>=(0..seen*1024).map(|v|2_000_000.+v as f32).collect();
        let kv:Vec<_>=(0..n*2048).map(|v|3_000_000.+v as f32).collect();
        for first in [0,8] {
            let x=group_history(&prefix,&keys,&values,&kv,n,total,first);
            assert_eq!(x.len(),2*2*total*256); assert!(x.len()<MAX_FLOATS);
            for part in 0..2 {for local in 0..2 {
                let head=first/4+local; let base=(part*2+local)*total*256;
                for token in 0..total {
                    let want=if token<PREFIX {let at=(part*4+head)*PREFIX*256+token*256;&prefix[at..at+256]}
                        else if token<PREFIX+seen {let v=if part==0{&keys}else{&values};let at=(token-PREFIX)*1024+head*256;&v[at..at+256]}
                        else {let at=part*n*1024+(token-PREFIX-seen)*1024+head*256;&kv[at..at+256]};
                    assert_eq!(&x[base+token*256..base+(token+1)*256],want);
                }
            }}
        }
    }
    #[test]
    fn invalid_scope_and_history_fail_before_weight_reads() {
        let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
        let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"attention_full_integer","tensor":"model.language_model.layers.3.self_attn.q_proj.weight","dims":[1,27,0],"scalars":[],"aux":[],"encoding":"bf16-block256-exact-v1"})).unwrap();
        let norm=vec![0.;C]; let prefix=vec![0.;PREFIX*2048];
        for dims in [vec![],vec![58,27,0],vec![1,512,0],vec![1,26,0],vec![1,27,1],vec![usize::MAX,27,0]] {
            r.dims=dims; assert!(evaluate(&r,&norm,&prefix,&[],&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("invalid read")}).is_err());
        }
        r.dims=vec![1,28,0];assert!(evaluate(&r,&norm,&prefix,&[],&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("missing history read")}).is_err());
        r.dims=vec![1,27,0];r.model="d".repeat(64);assert!(evaluate(&r,&norm,&prefix,&[],&[],&m,&mut|_,_|->Result<Vec<u8>>{panic!("scope read")}).is_err());
    }
}
