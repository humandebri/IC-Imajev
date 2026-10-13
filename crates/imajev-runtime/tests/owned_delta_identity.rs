#![cfg(feature = "experimental-prefix-hybrid")]
use imajev_runtime::{decode_query, evaluate_owned_decoded_with_prepared_buffer, Manifest, Request};
use sha2::{Digest, Sha256};

/// A prepared state must stay bound to every original request field even after
/// replacing JSON serialization with direct field comparisons.
#[test]
fn owned_operand_rejects_every_changed_identity_field_before_weight_reads() {
    let request: Request = serde_json::from_value(serde_json::json!({
        "version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),
        "input_hash":"c".repeat(64),"step":0,"op":"delta_full_hybrid_integer",
        "encoding":"delta-hybrid-prefix-exact-v1","dims":[1,32,2,0],
        "tensor":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "scalars":[]
    })).unwrap();
    let (packet, _) = imajev_runtime::prefix_hybrid_codec::prepare(&vec![0.; 2*6176], 2).unwrap();
    let header = serde_json::to_vec(&request).unwrap();
    let count = 2560 + 3*8192;
    let mut frame = (header.len() as u32).to_le_bytes().to_vec();
    frame.extend(header);
    frame.push(1);
    frame.extend((count as u32).to_le_bytes());
    frame.extend(vec![0; count*2]);
    frame.extend((packet.len() as u32).to_le_bytes());
    frame.extend(packet);
    let digest = Sha256::digest(&frame);
    frame.extend_from_slice(&digest);
    for field in 0..12 {
        let (mut changed, input) = decode_query(&frame).unwrap();
        match field {
            0 => changed.version = 2,
            1 => changed.model = "d".repeat(64),
            2 => changed.pack_hash = "e".repeat(64),
            3 => changed.input_hash = "f".repeat(64),
            4 => changed.step = 1,
            5 => changed.op = "delta_full_log_integer".into(),
            6 => changed.tensor = changed.tensor.replace("layers.0", "layers.1"),
            7 => changed.dims[2] = 3,
            8 => changed.scalars.push(-0.),
            9 => changed.aux.push("unexpected".into()),
            10 => changed.encoding = "bf16-block256-exact-v1".into(),
            11 => changed.dims[3] = 1,
            _ => unreachable!(),
        }
        // Matching the mutated model/pack forces identity validation as well.
        let manifest = Manifest { version:1, model:changed.model.clone(),
            pack_hash:changed.pack_hash.clone(), bytes:0, tensors:vec![] };
        let error = evaluate_owned_decoded_with_prepared_buffer(&changed, input, &manifest,
            |_,_| -> Result<Vec<u8>, String> { panic!("changed identity read weights") }).unwrap_err();
        assert_eq!(error, "hybrid Delta request identity", "field {field}");
    }
}
