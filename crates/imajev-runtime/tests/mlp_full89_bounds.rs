#![cfg(feature="experimental-mlp-full")]
use imajev_runtime::{Manifest,Request,evaluate_with_reader};
#[test]
fn expanded_token_bounds_require_the_explicit_feature() {
    let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:0,tensors:vec![]};
    for n in [88,89,90] {
        let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"mlp_full_integer","tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[n,2560],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"],"encoding":"bf16-block256-exact-v1"})).unwrap();
        let error=evaluate_with_reader(&r,&vec![0.;n*2560*2],&m,|_,_|panic!("must reject before reads")).unwrap_err();
        assert_eq!(error,if cfg!(feature="experimental-mlp-full89")&&n<=89 {"MLP pipeline missing weight"}else{"full MLP metadata/bounds"});
    }
}
