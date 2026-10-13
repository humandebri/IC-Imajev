#![cfg(feature="experimental-mlp-delta-fusion")]
use imajev_runtime::{Request,Manifest,decode_query,evaluate_decoded_with_prepared_buffer};
use sha2::{Digest,Sha256};
fn req()->Request {serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_finish_delta_log_integer","encoding":"mlp-delta-log-carry-exact-v1","tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[1,1],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"]})).unwrap()}
fn planar(raw:&[u8],width:usize)->Vec<u8>{(0..width).flat_map(|b|(0..raw.len()/width).map(move|i|raw[i*width+b])).collect()}
fn frame(r:&Request,streams:&[Vec<u8>])->Vec<u8>{let h=serde_json::to_vec(r).unwrap();let mut b=(h.len()as u32).to_le_bytes().to_vec();b.extend(h);b.push(1);if r.encoding=="mlp-delta-huffman-carry-exact-v1" {for (group,&width)in streams.iter().zip(&[2,1,4,2,2,4]){let count=group.len()/width;for plane in 0..width {b.push(0);b.extend((count as u32).to_le_bytes());b.extend(&group[plane*count..(plane+1)*count]);}}}else{let compressed:Vec<_>=streams.iter().map(|s|miniz_oxide::deflate::compress_to_vec_zlib(s,1)).collect();for c in &compressed{b.extend((c.len()as u32).to_le_bytes());}for c in compressed{b.extend(c);}}let sha=Sha256::digest(&b);b.extend(sha);b}
fn streams()->Vec<Vec<u8>>{let tail:Vec<_>=(0..100).flat_map(|i|if i<36{1f32.to_le_bytes()}else{0f32.to_le_bytes()}).collect();let mut log=vec![0;4096*4];for _ in 0..32{log.extend(0.5f32.to_le_bytes());}vec![vec![0;2560*2],vec![0;9216],planar(&tail,4),vec![0;24576*2],vec![0;2048*2],planar(&log,4)]}
fn check_identity(r:Request){let(_,input)=decode_query(&frame(&r,&streams())).unwrap();let mut changes=vec![];macro_rules! change{($field:ident,$value:expr)=>{let mut x=r.clone();x.$field=$value;changes.push(x);};}
 change!(version,2);change!(model,"d".repeat(64));change!(pack_hash,"e".repeat(64));change!(input_hash,"f".repeat(64));change!(step,1);change!(op,"mlp_finish_partial_integer".into());change!(encoding,"bf16-exact".into());change!(tensor,"model.language_model.layers.1.post_attention_layernorm.weight".into());change!(dims,vec![1,2]);change!(aux,vec!["model.language_model.layers.2.input_layernorm.weight".into()]);change!(scalars,vec![2.,f32::from_bits(1e-6f32.to_bits()+1)]);
 for altered in changes {let m=Manifest{version:1,model:altered.model.clone(),pack_hash:altered.pack_hash.clone(),bytes:0,tensors:vec![]};assert_eq!(evaluate_decoded_with_prepared_buffer(&altered,&input,&m,|_,_|->Result<Vec<u8>,String>{panic!("identity failure reached weights")}).unwrap_err(),"carry request identity");}
}
fn check_invalid(r:Request){let good=streams();assert!(decode_query(&frame(&r,&good)).is_ok());let mut bad=good.clone();bad[1][0]=128;assert!(decode_query(&frame(&r,&bad)).is_err());let mut bad=good.clone();let mut raw=vec![0;4096*4];for _ in 0..32{raw.extend(1.1f32.to_le_bytes());}bad[5]=planar(&raw,4);assert!(decode_query(&frame(&r,&bad)).is_err());for d in [vec![0,1],vec![90,1],vec![1,133],vec![usize::MAX,1]]{let mut bad=r.clone();bad.dims=d;assert!(decode_query(&frame(&bad,&good)).is_err());}let mut bad=r;bad.tensor="model.language_model.layers.2.post_attention_layernorm.weight".into();assert!(decode_query(&frame(&bad,&good)).is_err());}

#[test]fn every_request_field_is_bound_before_any_weight_read(){check_identity(req());}
#[test]fn decoder_rejects_bad_lanes_gates_and_metadata(){check_invalid(req());}
#[test]fn huffman_identity_before_weights(){let mut r=req();r.encoding="mlp-delta-huffman-carry-exact-v1".into();check_identity(r);}
#[test]fn huffman_invalid_values_and_shapes(){let mut r=req();r.encoding="mlp-delta-huffman-carry-exact-v1".into();check_invalid(r);}

#[test]fn fused_reply_uses_validated_shape_without_changing_normal_limits(){
 for encoding in ["mlp-delta-log-carry-exact-v1","mlp-delta-huffman-carry-exact-v1"] {
  for n in [87,89] {let mut r=req();r.dims=vec![n,45];r.encoding=encoding.into();let values=vec![0.;2*n*2560+24576];assert!(values.len()>450000 && values.len()<=imajev_runtime::MAX_FLOATS);let frame=imajev_runtime::encode(&r,&values).unwrap();assert_eq!(imajev_runtime::decode(&frame).unwrap().1,values);assert!(imajev_runtime::encode(&r,&values[..values.len()-1]).is_err());r.encoding=String::new();assert!(imajev_runtime::encode(&r,&values).is_err());r.dims[0]=90;r.encoding=encoding.into();assert!(imajev_runtime::encode(&r,&values).is_err());}
 }
}

#[test]fn partial_rows_are_explicit_validated_and_bound_to_request(){
 for rows in [32,256,512,2528] {
  let mut r=req();r.encoding="mlp-delta-huffman-carry-exact-v1".into();r.dims.push(rows);let(_,input)=decode_query(&frame(&r,&streams())).unwrap();let mut changed=r.clone();changed.dims[2]=if rows==512{256}else{512};let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};assert_eq!(evaluate_decoded_with_prepared_buffer(&changed,&input,&m,|_,_|->Result<Vec<u8>,String>{panic!("row identity reached weights")}).unwrap_err(),"carry request identity");
 }
 for rows in [0,1,31,33,2560,usize::MAX]{let mut r=req();r.dims.push(rows);assert!(decode_query(&frame(&r,&streams())).is_err());}
 let mut r=req();r.dims=vec![89,132,512];assert!(decode_query(&frame(&r,&streams())).is_err());
}
