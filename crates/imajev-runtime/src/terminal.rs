//! Fixed-model terminal MLP; preserves all separate-operation rounding boundaries.
use crate::{evaluate_integer, evaluate_mlp, execute, load_prepared_weight as load_weight, Manifest, Request, Result};
pub fn evaluate<F, B>(r: &Request, x: &[f32], m: &Manifest, read: &mut F) -> Result<(Vec<f32>, u64)>
where
    F: FnMut(u64, usize) -> Result<B>,
    B: crate::prepared_weights::WeightBuffer,
{
    const LAYER: &str = "model.language_model.layers.31";
    if r.dims != [1, 2560]
        || x.len() != 5120
        || !crate::lossless_encoding(&r.encoding)
        || !r.aux.is_empty()
        || !r.scalars.is_empty()
        || r.tensor != format!("{LAYER}.post_attention_layernorm.weight")
    {
        return Err("terminal MLP metadata".into());
    }
    let mut nr = r.clone();
    nr.op = "add_norm_bf16".into();
    nr.scalars = vec![1e-6];
    let tensor = m
        .tensors
        .iter()
        .find(|t| t.name == nr.tensor)
        .ok_or("terminal post norm")?;
    let (w, mut bytes) = load_weight(tensor, &nr, &mut *read)?;
    let residual = execute(&nr, x, &w)?;
    let mut mr = r.clone();
    mr.op = "mlp_gate_up_integer".into();
    mr.dims = vec![1, 9216, 2560, 0];
    mr.scalars = vec![2.];
    mr.tensor = format!("{LAYER}.mlp.gate_proj.weight");
    mr.aux = vec![format!("{LAYER}.mlp.up_proj.weight")];
    let (product, used) = evaluate_mlp(&mr, &residual[2560..], m, &mut *read)?;
    bytes += used;
    mr.op = "lora_integer".into();
    mr.dims = vec![1, 2560, 9216, 0];
    mr.tensor = format!("{LAYER}.mlp.down_proj.weight");
    mr.aux = vec![
        format!("{LAYER}.mlp.down_proj.lora_A.weight"),
        format!("{LAYER}.mlp.down_proj.lora_B.weight"),
    ];
    let (mlp, used) = evaluate_integer(&mr, &product, m, &mut *read, None)?;
    bytes += used;
    nr.tensor = "model.language_model.norm.weight".into();
    let tensor = m
        .tensors
        .iter()
        .find(|t| t.name == nr.tensor)
        .ok_or("terminal final norm")?;
    let (w, used) = load_weight(tensor, &nr, &mut *read)?;
    bytes += used;
    let input = [&residual[..2560], &mlp[..]].concat();
    Ok((execute(&nr, &input, &w)?, bytes))
}
