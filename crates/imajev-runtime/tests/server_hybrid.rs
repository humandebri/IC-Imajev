#![cfg(feature="experimental-prefix-hybrid")]
use imajev_runtime::{Request,server_delta_hybrid_input,DecodedQueryInput};
#[test]
fn server_input_keeps_checked_packet_contract() {
    let mut r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"delta_full_hybrid_integer","encoding":"delta-hybrid-prefix-exact-v1","tensor":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight","dims":[1,32,2,0],"scalars":[]})).unwrap();
    let(packet,_)=imajev_runtime::prefix_hybrid_codec::prepare(&vec![0.;2*6176],2).unwrap();
    let mut input=vec![0.;2560+3*8192];
    assert!(matches!(server_delta_hybrid_input(&r,&input,&packet).unwrap(),DecodedQueryInput::DeltaHybrid(_)));
    input[0]=f32::from_bits(1);assert!(server_delta_hybrid_input(&r,&input,&packet).is_err());
    input[0]=f32::INFINITY;assert!(server_delta_hybrid_input(&r,&input,&packet).is_err());
    input[0]=0.;r.dims[2]=3;assert!(server_delta_hybrid_input(&r,&input,&packet).is_err());
    r.dims[2]=2;assert!(server_delta_hybrid_input(&r,&input,&packet[..packet.len()-1]).is_err());
    input.pop();assert!(server_delta_hybrid_input(&r,&input,&packet).is_err());
}
