//! Experimental server-held standard text graph. No client supplied hidden state.
use super::*;
use imajev_runtime::Request;
const C: usize = 2560;
pub(super) const WORKER_BUDGET: u64 = 30_000_000_000;
pub(super) fn progress_limit(n:usize)->u64 {
    #[cfg(feature="experimental-update-token-chunks")]
    if n>89 {return crate::token_plan::stages(n);}
    let _=n;64
}
#[derive(Default)]
struct Prefix { values: Vec<f32>, packet: Vec<u8> }
#[derive(Default)]
struct GraphStore { prefix: Vec<Prefix>, session: Option<Session>, next_id: u64 }
struct Session {
    id: u64, stage: u64, n: usize, prefix_tokens: usize, options: Vec<String>, request: Request,
    hidden: Vec<f32>, norm: Vec<f32>, attention: Vec<f32>,
    hidden_hashes: Vec<String>, state_hashes: Vec<String>, decision: Option<ChoiceResult>,
}
#[derive(CandidType, Deserialize, Clone)]
pub(super) struct UpdateProgress {
    pub(super) id: u64, pub(super) stage: u64, pub(super) done: bool, pub(super) instructions: u64, pub(super) stable_read_bytes: u64,
    pub(super) operations: Vec<(String,u64)>, pub(super) hidden_hashes: Vec<String>, pub(super) state_hashes: Vec<String>,
    pub(super) final_hidden: Vec<f32>, pub(super) decision: Option<ChoiceResult>, pub(super) heap_pages: u64,
}
thread_local! {static GRAPH: RefCell<GraphStore> = RefCell::new(GraphStore::default());}
fn digest(v: &[f32]) -> String {
    let mut h=Sha256::new();for x in v {h.update(x.to_le_bytes());}
    h.finalize().iter().map(|b|format!("{b:02x}")).collect()
}
#[ic_cdk::update]
fn update_prefix(layer: u32, values: Vec<f32>, packet: StateBytes) -> Result<(),String> {
    owner();paid_admin_guard();let i=layer as usize;
    let p=validate_prefix(i,&values,&packet)?;
    if i%4!=3 {
        let r=template();let mut r=Request{op:"delta_full_hybrid_integer".into(),encoding:"delta-hybrid-prefix-exact-v1".into(),dims:vec![1,32,p,0],..r};
        r.tensor=format!("model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight");
        let mut input=vec![0.;C];input.extend_from_slice(&values);
        imajev_runtime::server_delta_hybrid_input(&r,&input,&packet)?;
    }
    GRAPH.with(|g| {let mut g=g.borrow_mut();if g.session.as_ref().is_some_and(|s|s.stage<64){return Err("prefix frozen after inference start".into());}
        if g.prefix.is_empty(){g.prefix=(0..32).map(|_|Prefix::default()).collect();}
        if !g.prefix[i].values.is_empty(){return Err("prefix already installed".into());}
        g.prefix[i]=Prefix{values,packet:packet.into_vec()};Ok(())})
}
fn validate_prefix(i:usize,values:&[f32],packet:&[u8])->Result<usize,String> {
    if i>=32 || !values.iter().all(|v|v.is_finite() && v.to_bits() & 65535 == 0) {return Err("prefix bounds/precision".into());}
    let p=if i%4==3 {
        if values.len()%2048!=0 {return Err("prefix KV shape".into());}
        values.len()/2048
    } else {
        if packet.len()<12 || &packet[..4]!=b"NPF1" {return Err("prefix Delta shape".into());}
        u32::from_le_bytes(packet[4..8].try_into().unwrap()) as usize
    };
    if p!=27 {return Err("prefix token bounds".into());}
    if i%4==3 {
        if values.len()!=p*2048 || !packet.is_empty() {return Err("prefix KV shape".into());}
    } else {
        if values.len()!=3*8192 || packet.len()<12 || &packet[..4]!=b"NPF1" || u32::from_le_bytes(packet[4..8].try_into().unwrap())!=p as u32 {return Err("prefix Delta shape".into());}
    }
    Ok(p)
}

fn template() -> Request {
    STORE.with(|s| {let s=s.borrow();let m=s.manifest.as_ref().expect("prepared manifest");Request{
        version:1,model:m.model.clone(),pack_hash:m.pack_hash.clone(),input_hash:"0".repeat(64),
        step:0,op:String::new(),tensor:String::new(),dims:vec![],scalars:vec![],aux:vec![],encoding:"bf16-block256-exact-v1".into()}})
}
#[ic_cdk::update]
fn update_infer_start(ids: Vec<u32>, options: Vec<String>) -> Result<UpdateProgress,String> {
    owner();paid_admin_guard();start(ids,options)
}
pub(super) fn start(ids: Vec<u32>, options: Vec<String>) -> Result<UpdateProgress,String> {
    if busy(){return Err("inference already active".into());}
    #[cfg(feature="experimental-update-token-chunks")]
    if ids.len()>89 {return crate::chunked_update::start(ids,options);}
    if !(1..=89).contains(&ids.len()){return Err("suffix token bounds".into());}
    STORE.with(|s|if s.borrow().ready {Ok(())}else{Err("not ready".to_string())})?;
    imajev_runtime::decide_candidates(&options,&vec![0.;options.len().min(7)+1],1.3051569717552742)?;
    GRAPH.with(|g| -> Result<(),String> {let g=g.borrow();if g.prefix.len()!=32 || g.prefix.iter().any(|p|p.values.is_empty()){return Err("prefix incomplete".into());}
        if g.session.as_ref().is_some_and(|s|s.stage<64){return Err("inference already active".into());}Ok(())})?;
    let mut r=template();let n=ids.len();r.op="embed".into();r.tensor="model.language_model.embed_tokens.weight".into();r.dims=vec![n,C];
    let started=ic_cdk::api::performance_counter(0);let before=started;
    let (hidden,mut reads)=evaluate(&r,&ids.iter().map(|v|*v as f32).collect::<Vec<_>>()).unwrap_or_else(|e|ic_cdk::trap(&e));
    let embedding=ic_cdk::api::performance_counter(0)-before;
    r.op="rms_bf16".into();r.tensor="model.language_model.layers.0.input_layernorm.weight".into();r.scalars=vec![1e-6];
    let before=ic_cdk::api::performance_counter(0);let(norm,b)=evaluate(&r,&hidden).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;
    let initial_norm=ic_cdk::api::performance_counter(0)-before;
    let id=GRAPH.with(|g| {let mut g=g.borrow_mut();g.next_id=g.next_id.checked_add(1).expect("session id overflow");g.next_id});
    let prefix_tokens=27;
    let s=Session{id,stage:0,n,prefix_tokens,options,request:r,hidden,norm,attention:vec![],hidden_hashes:vec![],state_hashes:vec![],decision:None};
    Ok(run(s,started,reads,vec![("embed".into(),embedding),("initial_norm".into(),initial_norm)]))
}
#[ic_cdk::update]
fn update_infer_continue(id:u64, stage:u64) -> Result<UpdateProgress,String> {
    owner();paid_admin_guard();continue_graph(id,stage)
}
pub(super) fn continue_graph(id:u64,stage:u64) -> Result<UpdateProgress,String> {
    #[cfg(feature="experimental-update-token-chunks")]
    if crate::chunked_update::busy() {return crate::chunked_update::continue_graph(id,stage);}
    GRAPH.with(|g| -> Result<(),String> {let g=g.borrow();let s=g.session.as_ref().ok_or("missing session")?;
        if s.id!=id || s.stage!=stage || stage>=64 {return Err("session progress mismatch".into());}Ok(())})?;
    let start=ic_cdk::api::performance_counter(0);
    let s=GRAPH.with(|g|g.borrow_mut().session.take().unwrap());Ok(run(s,start,0,vec![]))
}
fn run(mut s:Session,start:u64,mut reads:u64,mut ops:Vec<(String,u64)>) -> UpdateProgress {
    while s.stage<64 && ic_cdk::api::performance_counter(0)-start<WORKER_BUDGET {
        let layer=(s.stage/2)as usize;let root=format!("model.language_model.layers.{layer}");
        let before=ic_cdk::api::performance_counter(0);let mut r=s.request.clone();r.step=s.stage;r.aux.clear();r.scalars.clear();
        if s.stage%2==0 {
            if layer%4==3 {
                r.op="attention_full_integer".into();r.tensor=format!("{root}.self_attn.q_proj.weight");r.dims=vec![s.n,s.prefix_tokens,usize::from(layer==31)];
                let mut x=s.norm.clone();GRAPH.with(|g|x.extend_from_slice(&g.borrow().prefix[layer].values));
                let(y,b)=evaluate(&r,&x).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;
                let count=if layer==31 {C}else{s.n*C};s.state_hashes.push(digest(&y[count..]));s.attention=y[..count].to_vec();
                if layer==31 {s.hidden=s.hidden[s.hidden.len()-C..].to_vec();s.n=1;}
            } else {
                r.op="delta_full_hybrid_integer".into();r.tensor=format!("{root}.linear_attn.in_proj_qkv.weight");r.dims=vec![s.n,32,s.prefix_tokens,0];r.encoding="delta-hybrid-prefix-exact-v1".into();
                let input=GRAPH.with(|g| {let g=g.borrow();let p=&g.prefix[layer];let mut x=s.norm.clone();x.extend_from_slice(&p.values);
                    imajev_runtime::server_delta_hybrid_input(&r,&x,&p.packet).unwrap_or_else(|e|ic_cdk::trap(&e))});
                let(y,b)=evaluate_decoded(&r,input).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;
                let y=y.into_values().unwrap_or_else(|e|ic_cdk::trap(&e));s.state_hashes.push(digest(&y[s.n*C..]));s.attention=y[..s.n*C].to_vec();
            }
        } else {
            r.op="mlp_full_integer".into();r.tensor=format!("{root}.post_attention_layernorm.weight");r.dims=vec![s.n,C];r.scalars=vec![2.,1e-6];
            r.aux=vec![if layer==31 {"model.language_model.norm.weight".into()}else{format!("model.language_model.layers.{}.input_layernorm.weight",layer+1)}];
            let mut x=std::mem::take(&mut s.hidden);x.extend_from_slice(&s.attention);
            let(y,b)=evaluate(&r,&x).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;
            s.hidden=y[..s.n*C].to_vec();s.norm=y[s.n*C..].to_vec();s.attention.clear();s.hidden_hashes.push(digest(&s.hidden));
        }
        ops.push((r.op,ic_cdk::api::performance_counter(0)-before));s.stage+=1;
    }
    if s.stage==64 {
        let before=ic_cdk::api::performance_counter(0);let mut r=s.request.clone();r.op="matmul".into();r.tensor="readout-f32".into();r.dims=vec![1,s.options.len()+1,C,0];r.aux.clear();r.scalars.clear();
        let(logits,b)=evaluate(&r,&s.norm).unwrap_or_else(|e|ic_cdk::trap(&e));reads+=b;
        let d=imajev_runtime::decide_candidates(&s.options,&logits,1.3051569717552742).unwrap_or_else(|e|ic_cdk::trap(&e));
        s.decision=Some(ChoiceResult{value:d.value,probabilities:d.probabilities,unknown_probability:d.unknown_probability,abstained:d.abstained,raw_logits:d.raw_logits,
            instructions:ic_cdk::api::performance_counter(0)-before,calibration_version:"p3-r2-s000291-authored".into()});
    }
    #[cfg(target_arch="wasm32")]
    let heap_pages=core::arch::wasm32::memory_size(0) as u64;
    #[cfg(not(target_arch="wasm32"))]
    let heap_pages=0;
    let reply=UpdateProgress{id:s.id,stage:s.stage,done:s.stage==64,instructions:ic_cdk::api::performance_counter(0)-start,stable_read_bytes:reads,operations:ops,
        hidden_hashes:s.hidden_hashes.clone(),state_hashes:s.state_hashes.clone(),final_hidden:if s.stage==64{s.norm.clone()}else{vec![]},decision:s.decision.clone(),
        heap_pages};
    GRAPH.with(|g|g.borrow_mut().session=Some(s));reply
}

pub(super) fn bank_ready(tokens:usize)->bool {tokens==27 && GRAPH.with(|g|{let g=g.borrow();g.prefix.len()==32 && g.prefix.iter().all(|p|!p.values.is_empty())})}
pub(super) fn select_bank(tokens:usize)->Result<(),String> {if bank_ready(tokens){Ok(())}else{Err("prefix incomplete".into())}}
pub(super) fn busy()->bool {
    #[cfg(feature="experimental-update-token-chunks")]
    if crate::chunked_update::busy(){return true;}
    GRAPH.with(|g|g.borrow().session.as_ref().is_some_and(|s|s.stage<64))
}
pub(super) fn release(){
    #[cfg(feature="experimental-update-token-chunks")]
    crate::chunked_update::release();
    GRAPH.with(|g|g.borrow_mut().session=None);
}
#[cfg(feature="experimental-update-token-chunks")]
pub(super) fn stream_prefix(layer:usize)->Result<(Vec<f32>,Option<imajev_runtime::ServerDeltaStream>),String> {
    GRAPH.with(|g|{
        let g=g.borrow();let p=g.prefix.get(layer).ok_or("missing prefix layer")?;
        if p.values.is_empty(){return Err("prefix incomplete".into());}
        let stream=if layer%4==3 {None}else{
            let mut r=template();r.op="delta_full_hybrid_integer".into();r.encoding="delta-hybrid-prefix-exact-v1".into();
            r.tensor=format!("model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight");r.dims=vec![1,32,27,0];
            let mut x=vec![0.;C];x.extend_from_slice(&p.values);
            Some(imajev_runtime::server_delta_stream(&r,&x,&p.packet)?)
        };
        Ok((p.values.clone(),stream))
    })
}
fn paid_admin_guard(){#[cfg(feature="paid-update-inference")]crate::paid_inference::admin_guard();}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registration_rejects_voting_bank_for_attention_and_delta() {
        assert_eq!(validate_prefix(3,&vec![0.;27*2048],&[]).unwrap(),27);
        assert!(validate_prefix(3,&vec![0.;38*2048],&[]).is_err());
        let mut packet=b"NPF1".to_vec();packet.extend(27u32.to_le_bytes());packet.extend([0;4]);
        assert_eq!(validate_prefix(0,&vec![0.;3*8192],&packet).unwrap(),27);
        packet[4..8].copy_from_slice(&38u32.to_le_bytes());
        assert!(validate_prefix(0,&vec![0.;3*8192],&packet).is_err());
    }
    #[test]
    fn readiness_requires_every_common_layer() {
        GRAPH.with(|g|*g.borrow_mut()=GraphStore::default());
        assert!(!bank_ready(27));
        GRAPH.with(|g|g.borrow_mut().prefix=(0..32).map(|_|Prefix{values:vec![0.],packet:vec![]}).collect());
        assert!(bank_ready(27));assert!(!bank_ready(38));
        GRAPH.with(|g|g.borrow_mut().prefix[31].values.clear());
        assert!(!bank_ready(27));
    }
}
