//! Experimental layer-by-layer token tiles; all carry is owned by one job.
use super::*;
use imajev_runtime::{Request, ServerDeltaStream};
const CHUNK:usize=57;
const C:usize=2560;
const PREFIX:usize=super::paid_inference::COMMON_PREFIX.len();
struct Session {
    id:u64, stage:u64, n:usize, plan:Vec<crate::token_plan::Tile>, request:Request, options:Vec<String>,
    hidden:Vec<f32>, norm:Vec<f32>, attention:Vec<f32>,
    conv:Vec<f32>, delta:Option<ServerDeltaStream>, prefix_kv:Vec<f32>, keys:Vec<f32>, values:Vec<f32>,
    hidden_hashes:Vec<String>, state_hashes:Vec<String>, decision:Option<ChoiceResult>,
}
thread_local! {
    static SESSION:RefCell<Option<Session>>=RefCell::new(None);
    static NEXT_ID:RefCell<u64>=RefCell::new(0);
}
pub(super) fn busy()->bool {SESSION.with(|s|s.borrow().as_ref().is_some_and(|v|v.stage<v.plan.len() as u64))}
pub(super) fn release(){SESSION.with(|s|*s.borrow_mut()=None);}
fn digest(v:&[f32])->String {
    let mut h=Sha256::new();for x in v {h.update(x.to_le_bytes());}
    h.finalize().iter().map(|v|format!("{v:02x}")).collect()
}
fn template()->Result<Request,String> {
    STORE.with(|s|{let s=s.borrow();if !s.ready{return Err("not ready".into());}
        let m=s.manifest.as_ref().ok_or("missing manifest")?;
        Ok(Request{version:1,model:m.model.clone(),pack_hash:m.pack_hash.clone(),input_hash:"0".repeat(64),
                   step:0,op:String::new(),tensor:String::new(),dims:vec![],scalars:vec![],aux:vec![],encoding:"bf16-block256-exact-v1".into()})})
}
pub(super) fn start(ids:Vec<u32>,options:Vec<String>)->Result<UpdateProgress,String> {
    if !(90..=super::paid_inference::INPUT_LIMIT-PREFIX).contains(&ids.len()) || ids.iter().any(|&v|v>=248320){return Err("chunked suffix bounds".into());}
    if super::update_inference::busy(){return Err("inference already active".into());}
    if !super::update_inference::bank_ready(PREFIX){return Err("prefix incomplete".into());}
    imajev_runtime::decide_candidates(&options,&vec![0.;options.len().min(7)+1],1.3051569717552742)?;
    let start=ic_cdk::api::performance_counter(0);
    let mut r=template()?;let n=ids.len();r.op="embed".into();r.tensor="model.language_model.embed_tokens.weight".into();r.dims=vec![n,C];
    let mut hidden=Vec::with_capacity(n*C);let mut reads=0;
    for tile in ids.chunks(CHUNK) {
        r.dims=vec![tile.len(),C];
        let(y,b)=evaluate(&r,&tile.iter().map(|&v|v as f32).collect::<Vec<_>>())?;reads+=b;hidden.extend(y);
    }
    r.op="rms_bf16".into();r.tensor="model.language_model.layers.0.input_layernorm.weight".into();r.scalars=vec![1e-6];
    let mut norm=Vec::with_capacity(n*C);
    for tile in hidden.chunks(CHUNK*C) {r.dims=vec![tile.len()/C,C];let(y,b)=evaluate(&r,tile)?;reads+=b;norm.extend(y);}
    let id=NEXT_ID.with(|v|{let mut v=v.borrow_mut();*v=v.checked_add(1).expect("stream session ID overflow");*v});
    let s=Session{id,stage:0,n,plan:crate::token_plan::schedule(n),request:r,options,hidden,norm,attention:vec![],conv:vec![],delta:None,
                  prefix_kv:vec![],keys:vec![],values:vec![],hidden_hashes:vec![],state_hashes:vec![],decision:None};
    run(s,start,reads)
}
pub(super) fn continue_graph(id:u64,stage:u64)->Result<UpdateProgress,String> {
    SESSION.with(|v| -> Result<(),String> {let v=v.borrow();let s=v.as_ref().ok_or("missing stream session")?;
        if s.id!=id || s.stage!=stage || s.stage>=s.plan.len() as u64 {return Err("stream progress mismatch".into());}Ok(())})?;
    let start=ic_cdk::api::performance_counter(0);
    let s=SESSION.with(|v|v.borrow_mut().take().unwrap());run(s,start,0)
}
fn delta_tile(s:&mut Session,r:&Request,x:&[f32],offset:usize)->Result<(Vec<f32>,u64),String> {
    let stream=s.delta.as_mut().ok_or("missing Delta stream")?;
    STORE.with(|store|{let store=store.borrow();let m=store.manifest.as_ref().ok_or("missing manifest")?;
        let mut physical=0;
        let(out,_)=stream.evaluate(r,x,offset,m,&mut |at,len|{
            if at.checked_add(len as u64).is_none_or(|end|end>m.bytes){return Err("stream weight range".into());}
            if let Some(v)=store.weight_cache.read(at,len){return Ok(v);}
            let mut v=vec![0;len];ic_cdk::api::stable_read(at,&mut v);physical+=len as u64;
            Ok(weight_cache::ReadBuffer::Owned(v))
        })?;Ok((out,physical))})
}
// attention_full expects [K,V][head][token][dimension]; returned KV is token-major.
fn attention_tile(s:&Session,r:&Request,norm:&[f32])->Result<(Vec<f32>,u64),String> {
    STORE.with(|store|{let store=store.borrow();let m=store.manifest.as_ref().ok_or("missing manifest")?;
        let mut physical=0;
        let(out,_)=imajev_runtime::server_attention_token_tile(r,norm,&s.prefix_kv,&s.keys,&s.values,m,&mut|at,len|{
            if at.checked_add(len as u64).is_none_or(|end|end>m.bytes){return Err("Attention weight range".into());}
            if let Some(v)=store.weight_cache.read(at,len){return Ok(v);}
            let mut v=vec![0;len];ic_cdk::api::stable_read(at,&mut v);physical+=len as u64;
            Ok(weight_cache::ReadBuffer::Owned(v))
        })?;Ok((out,physical))})
}
fn history(s:&Session)->Vec<f32> {
    let seen=s.keys.len()/1024;
    let mut out=Vec::with_capacity((PREFIX+seen)*2048);
    for part in 0..2 {let suffix=if part==0 {&s.keys}else{&s.values};
        for head in 0..4 {
            let at=part*PREFIX*1024+head*PREFIX*256;
            out.extend_from_slice(&s.prefix_kv[at..at+PREFIX*256]);
            for token in 0..seen {let at=token*1024+head*256;out.extend_from_slice(&suffix[at..at+256]);}
        }
    }out
}
fn run(mut s:Session,start:u64,mut reads:u64)->Result<UpdateProgress,String> {
    let end=s.plan.len() as u64;let mut operations=vec![];
    while s.stage<end && ic_cdk::api::performance_counter(0)-start<super::paid_inference::worker_budget() {
        let tile=s.plan[s.stage as usize];let layer=tile.layer;let phase=usize::from(tile.mlp);
        let begin=tile.begin;let count=tile.count;
        let root=format!("model.language_model.layers.{layer}");
        let before=ic_cdk::api::performance_counter(0);
        let mut r=s.request.clone();r.step=s.stage;r.aux.clear();r.scalars.clear();
        if phase==0 {
            if tile.first {
                let(values,stream)=super::update_inference::stream_prefix(layer)?;
                if layer%4==3 {s.prefix_kv=values;}else{s.conv=values;s.delta=stream;}
                s.attention=vec![0.;if layer==31 {C}else{s.n*C}];
            }
            if layer%4==3 {
                r.op="attention_full_integer".into();r.tensor=format!("{root}.self_attn.q_proj.weight");
                r.dims=vec![count,PREFIX+begin,usize::from(layer==31)];
                let(y,b)=if s.n>229 && imajev_runtime::server_attention_needs_split(count,PREFIX+begin,layer==31) {
                    attention_tile(&s,&r,&s.norm[begin*C..(begin+count)*C])?
                }else{
                    let mut x=s.norm[begin*C..(begin+count)*C].to_vec();x.extend(history(&s));evaluate(&r,&x)?
                };reads+=b;
                let result_count=if layer==31 {C}else{count*C};
                if layer==31 {s.attention.copy_from_slice(&y[..C]);}else{s.attention[begin*C..(begin+count)*C].copy_from_slice(&y[..result_count]);}
                s.keys.extend_from_slice(&y[result_count..result_count+count*1024]);
                s.values.extend_from_slice(&y[result_count+count*1024..]);
            }else{
                r.op="delta_full_hybrid_integer".into();r.encoding="delta-hybrid-prefix-exact-v1".into();
                r.tensor=format!("{root}.linear_attn.in_proj_qkv.weight");r.dims=vec![count,32,PREFIX,0];
                let mut x=s.norm[begin*C..(begin+count)*C].to_vec();x.extend_from_slice(&s.conv);
                let(y,b)=delta_tile(&mut s,&r,&x,begin)?;reads+=b;
                s.attention[begin*C..(begin+count)*C].copy_from_slice(&y[..count*C]);s.conv=y[count*C..].to_vec();
            }
            if tile.last {
                if layer%4==3 {s.keys.append(&mut s.values);s.state_hashes.push(digest(&s.keys));s.keys.clear();s.prefix_kv.clear();}
                else{s.state_hashes.push(digest(&s.conv));s.delta=None;s.conv.clear();}
            }
        }else{
            let(begin,count)=if layer==31 {(0,1)}else{(begin,count)};
            if layer==31 {s.hidden=s.hidden[s.hidden.len()-C..].to_vec();s.norm.resize(C,0.);}
            r.op="mlp_full_integer".into();r.tensor=format!("{root}.post_attention_layernorm.weight");r.dims=vec![count,C];r.scalars=vec![2.,1e-6];
            r.aux=vec![if layer==31 {"model.language_model.norm.weight".into()}else{format!("model.language_model.layers.{}.input_layernorm.weight",layer+1)}];
            let mut x=s.hidden[begin*C..(begin+count)*C].to_vec();x.extend_from_slice(&s.attention[begin*C..(begin+count)*C]);
            let(y,b)=evaluate(&r,&x)?;reads+=b;
            s.hidden[begin*C..(begin+count)*C].copy_from_slice(&y[..count*C]);
            s.norm[begin*C..(begin+count)*C].copy_from_slice(&y[count*C..]);
            if tile.last {s.hidden_hashes.push(digest(&s.hidden));s.attention.clear();}
        }
        operations.push((r.op,ic_cdk::api::performance_counter(0)-before));
        s.stage+=1;
    }
    if s.stage==end {
        let before=ic_cdk::api::performance_counter(0);let mut r=s.request.clone();r.op="matmul".into();r.tensor="readout-f32".into();
        r.dims=vec![1,s.options.len()+1,C,0];r.aux.clear();r.scalars.clear();
        let(logits,b)=evaluate(&r,&s.norm)?;reads+=b;
        let d=imajev_runtime::decide_candidates(&s.options,&logits,1.3051569717552742)?;
        s.decision=Some(ChoiceResult{value:d.value,probabilities:d.probabilities,unknown_probability:d.unknown_probability,abstained:d.abstained,
            raw_logits:d.raw_logits,instructions:ic_cdk::api::performance_counter(0)-before,calibration_version:"p3-r2-s000291-authored".into()});
    }
    #[cfg(target_arch="wasm32")]
    let heap_pages=core::arch::wasm32::memory_size(0)as u64;
    #[cfg(not(target_arch="wasm32"))]
    let heap_pages=0;
    let reply=UpdateProgress{id:s.id,stage:s.stage,done:s.stage==end,instructions:ic_cdk::api::performance_counter(0)-start,stable_read_bytes:reads,
        operations,hidden_hashes:s.hidden_hashes.clone(),state_hashes:s.state_hashes.clone(),final_hidden:if s.stage==end{s.norm.clone()}else{vec![]},
        decision:s.decision.clone(),heap_pages};
    SESSION.with(|v|*v.borrow_mut()=Some(s));Ok(reply)
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn kv_history_preserves_prefix_and_suffix_layout(){
        let mut s=Session{id:0,stage:0,n:90,plan:crate::token_plan::schedule(90),request:serde_json::from_value(serde_json::json!({"version":1,"model":"a","pack_hash":"b","input_hash":"c","step":0,"op":"x","tensor":"x","dims":[],"scalars":[]})).unwrap(),options:vec![],hidden:vec![],norm:vec![],attention:vec![],conv:vec![],delta:None,prefix_kv:(0..PREFIX*2048).map(|i|i as f32).collect(),keys:vec![],values:vec![],hidden_hashes:vec![],state_hashes:vec![],decision:None};
        assert_eq!(history(&s),s.prefix_kv);
        s.keys=(0..2048).map(|i|100_000.+i as f32).collect();s.values=(0..2048).map(|i|200_000.+i as f32).collect();
        let h=history(&s);
        for part in 0..2 {for head in 0..4 {let out=(part*4+head)*(PREFIX+2)*256;
            assert_eq!(&h[out..out+PREFIX*256],&s.prefix_kv[(part*4+head)*PREFIX*256..(part*4+head+1)*PREFIX*256]);
            let v=if part==0{&s.keys}else{&s.values};for t in 0..2{assert_eq!(&h[out+PREFIX*256+t*256..out+PREFIX*256+(t+1)*256],&v[t*1024+head*256..t*1024+(head+1)*256]);}
        }}
    }
}
