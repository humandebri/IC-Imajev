#![cfg(feature="experimental-lora-input-sharing")]
use imajev_runtime::{Request,Tensor,Manifest,evaluate_with_reader,execute};

#[test]
fn shared_rank64_mlp_matches_independent_projections_and_read_bytes() {
    let mut pack=Vec::new();let mut tensors=Vec::new();
    for (which,kind) in ["gate","up"].iter().enumerate() {
        let root=format!("model.language_model.layers.0.mlp.{kind}_proj");
        let offset=pack.len()as u64;
        pack.extend((0..8*256).map(|i|((i*(which+3)%19)as i8-9)as u8));
        for _ in 0..8 {pack.extend(0.003f32.to_le_bytes());}
        tensors.push(Tensor{name:format!("{root}.weight"),offset,rows:8,cols:256,dtype:"int8".into(),bytes:8*260});
        for (suffix,rows,cols) in [("lora_A",64,256),("lora_B",8,64)] {
            let offset=pack.len()as u64;
            for i in 0..rows*cols {pack.extend((((i*(which+7)%53)as f32-26.)/1000.).to_le_bytes());}
            tensors.push(Tensor{name:format!("{root}.{suffix}.weight"),offset,rows,cols,dtype:"f32".into(),bytes:(rows*cols*4)as u64});
        }
    }
    let m=Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:pack.len()as u64,tensors};
    let read=|offset:u64,len:usize|pack.get(offset as usize..offset as usize+len).map(|s|s.to_vec()).ok_or("test bounds".into());
    for n in [7,32,45,80,87,89,132] {
        let x:Vec<f32>=(0..n*256).map(|i|((i*13%47)as f32-23.)/16.).collect();
        let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"mlp_gate_up_integer","tensor":"model.language_model.layers.0.mlp.gate_proj.weight","dims":[n,8,256,0],"scalars":[2.],"aux":["model.language_model.layers.0.mlp.up_proj.weight"]})).unwrap();
        let (got,bytes)=evaluate_with_reader(&r,&x,&m,read).unwrap();let mut both=Vec::new();let mut independent_bytes=0;
        for name in [&r.tensor,&r.aux[0]] {
            let mut single=r.clone();single.op="lora_integer".into();single.tensor=name.clone();
            let root=name.strip_suffix(".weight").unwrap();single.aux=vec![format!("{root}.lora_A.weight"),format!("{root}.lora_B.weight")];
            let (values,used)=evaluate_with_reader(&single,&x,&m,read).unwrap();both.extend(values);independent_bytes+=used;
        }
        let mut sw=r.clone();sw.op="swiglu_bf16".into();sw.dims=vec![n*8];sw.scalars.clear();
        let expected=execute(&sw,&both,&[]).unwrap();
        assert!(got.iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()),"n={n}");
        assert_eq!(bytes,independent_bytes);
    }
}
