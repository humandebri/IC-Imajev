//! Caller-funded inference. One active graph; each heavy batch is a new self-call.
use super::*;
use super::paid_types::*;
use ic_cdk::call::Call;
use serde::{Serialize,Deserialize};
const RECEIPTS:usize=128;
const TTL:u64=86_400_000_000_000;
const MAX_STEPS:u32=6;
const PREFIX27:[u32;27]=[248045,846,198,56555,279,2420,5721,321,4087,279,3296,1608,279,10661,12521,13,3301,1132,279,3074,2904,1970,13,198,1349,25,328];
#[derive(Serialize,Deserialize)]
struct PaidStore {config:Config,next_id:u64,receipts:Vec<Receipt>,active:Option<Job>}
#[derive(Serialize,Deserialize)]
struct Job {id:u64,stage:u64,graph_id:Option<u64>,request:InferRequest,quote:Quote,in_flight:bool,workers:Vec<WorkerMetric>}
impl Default for PaidStore {fn default()->Self {Self{config:Config{enabled:false,version:1,fee_per_token:0,base_fee:0,reserve_cycles:0},next_id:0,receipts:vec![],active:None}}}
thread_local! {static PAID:RefCell<PaidStore>=RefCell::new(PaidStore::default());static DEBUG:RefCell<Option<DebugResult>>=RefCell::new(None);}
#[cfg(feature="paid-update-diagnostics")]
thread_local! {static FAULT:RefCell<Option<(u64,bool)>>=RefCell::new(None);static REFUND_FAULT:RefCell<bool>=RefCell::new(false);}
fn decision(d:ChoiceResult)->Decision {Decision{value:d.value,probabilities:d.probabilities,unknown_probability:d.unknown_probability,abstained:d.abstained,raw_logits:d.raw_logits,instructions:d.instructions,calibration_version:d.calibration_version}}
fn suffix_tokens(r:&InferRequest)->Result<usize,InferError> {
 if r.version!=1 || r.model.len()!=64 || r.token_ids.len()>84 || r.token_ids.is_empty() || r.token_ids.iter().any(|i|*i>=248320) {return Err(InferError::Invalid("input bounds/version".into()));}
 if !r.token_ids.starts_with(&PREFIX27){return Err(InferError::Invalid("common prefix mismatch".into()));}
 let n=r.token_ids.len()-PREFIX27.len();
 if !(1..=57).contains(&n){return Err(InferError::Invalid("suffix bounds".into()));}
 Ok(n)
}
fn plan(r:&InferRequest)->Result<Quote,InferError> {
 let n=suffix_tokens(r)?;
 imajev_runtime::decide_candidates(&r.options,&vec![0.;r.options.len().min(7)+1],1.3051569717552742).map_err(InferError::Invalid)?;
 let valid=STORE.with(|s|s.borrow().manifest.as_ref().is_some_and(|m|m.model==r.model));
 if !valid {return Err(InferError::Invalid("model mismatch".into()));}
 let p=PREFIX27.len();
 if !update_inference::bank_ready(p){return Err(InferError::NotReady);}
 if !STORE.with(|s|s.borrow().ready && s.borrow().weight_cache.names().len()==721) {return Err(InferError::NotReady);}
 PAID.with(|s|{let s=s.borrow();let fee=s.config.fee_per_token.checked_mul(n as u128).and_then(|v|v.checked_add(s.config.base_fee)).ok_or_else(||InferError::Invalid("fee overflow".into()))?;
  if fee==0{return Err(InferError::NotReady);}
  Ok(Quote{version:s.config.version,fee,prefix_tokens:p as u32,suffix_tokens:n as u32,estimated_steps:((n as u64*2_600_000_000).div_ceil(34_000_000_000) as u32),max_steps:MAX_STEPS})})
}
#[ic_cdk::query]
fn quote(request:InferRequest)->Result<Quote,InferError>{plan(&request)}
#[ic_cdk::update]
fn configure_paid(config:Config)->Result<(),String>{owner_auth();apply_config(config)}
fn apply_config(config:Config)->Result<(),String>{if config.version==0 || config.base_fee>1_000_000_000_000_000 || config.fee_per_token>1_000_000_000_000 || config.reserve_cycles>1_000_000_000_000_000 || (config.enabled && config.base_fee==0){return Err("config bounds".into());}
 PAID.with(|s|{let mut s=s.borrow_mut();if s.active.is_some() && config.enabled{return Err("paid inference active".into());}if config.version<=s.config.version{return Err("version must increase".into());}s.config=config;Ok(())})}
fn input_hash(request:&InferRequest)->String {let b=candid::encode_one(request).expect("validated request encoding");Sha256::digest(b).iter().map(|b|format!("{b:02x}")).collect()}
fn prune(s:&mut PaidStore){let now=ic_cdk::api::time();s.receipts.retain(|r|matches!(r.state,ReceiptState::Running) || !matches!(r.refund,Refund::None|Refund::Done) || now.saturating_sub(r.time)<TTL);}
#[ic_cdk::update]
async fn infer(request:InferRequest,request_id:String,quote_version:u64)->Result<InferenceResult,InferError> {
 let caller=ic_cdk::api::msg_caller();
 if caller==Principal::anonymous() || caller==ic_cdk::api::canister_self() || request_id.is_empty() || request_id.len()>64 {return Err(InferError::Invalid("caller/request ID".into()));}
 // Preserve replay of pre-upgrade receipts with up to 95 tokens; plan limits new work to 84.
 if request.token_ids.len()>95 || request.options.len()>7 || request.options.iter().any(|s|s.len()>128){return Err(InferError::Invalid("input size".into()));}
 let hash=input_hash(&request);
 let duplicate=PAID.with(|s|{let mut s=s.borrow_mut();prune(&mut s);s.receipts.iter().find(|r|r.caller==caller && r.request_id==request_id).cloned()});
 if let Some(r)=duplicate {if r.input_hash!=hash{return Err(InferError::IdConflict);}return match r.state {ReceiptState::Completed(v)=>Ok(v),ReceiptState::Running=>Err(InferError::InProgress{job_id:r.job_id}),ReceiptState::Failed(reason)=>Err(InferError::Failed{job_id:r.job_id,reason,refund:r.refund})};}
 let q=plan(&request)?;
 if q.version!=quote_version{return Err(InferError::QuoteChanged{current:q.version});}
 let id=PAID.with(|s|->Result<u64,InferError>{let mut s=s.borrow_mut();if !s.config.enabled{return Err(InferError::Paused);}if s.active.is_some() || update_inference::busy(){return Err(InferError::Busy);}if s.receipts.len()>=RECEIPTS{return Err(InferError::ReceiptCapacity);}
  if ic_cdk::api::msg_cycles_available()<q.fee{return Err(InferError::InsufficientCycles{required:q.fee});}
  if ic_cdk::api::canister_liquid_cycle_balance()<s.config.reserve_cycles.saturating_add(q.fee){return Err(InferError::NotReady);}
  let id=s.next_id.checked_add(1).ok_or(InferError::ReceiptCapacity)?;
  let accepted=ic_cdk::api::msg_cycles_accept(q.fee);assert_eq!(accepted,q.fee,"cycles acceptance");s.next_id=id;
  s.receipts.push(Receipt{caller,request_id:request_id.clone(),input_hash:hash,job_id:id,quote:q.clone(),time:ic_cdk::api::time(),state:ReceiptState::Running,refund:Refund::None});
  s.active=Some(Job{id,stage:0,graph_id:None,request,quote:q.clone(),in_flight:false,workers:vec![]});Ok(id)})?;
 loop {
  let stage=PAID.with(|s|{let mut s=s.borrow_mut();let j=s.active.as_mut().expect("active job");assert_eq!(j.id,id);j.in_flight=true;j.stage});
  let response=Call::unbounded_wait(ic_cdk::api::canister_self(),"inference_step").with_args(&(id,stage)).await;
  let decoded=response.map_err(|e|format!("worker call: {e}")).and_then(|v|v.candid::<Result<StepResult,String>>().map_err(|e|format!("worker decode: {e}")).and_then(|v|v));
  match decoded {
   Ok(v) if v.job_id==id && v.stage>stage && v.stage<=64 => {
    let status=PAID.with(|s|{let mut s=s.borrow_mut();if let Some(j)=s.active.as_mut(){if j.id!=id || j.stage!=v.stage || !j.in_flight{return None;}j.in_flight=false;}
      s.receipts.iter().find(|r|r.job_id==id).map(|r|r.state.clone())});
    if v.done {if let Some(ReceiptState::Completed(result))=status {PAID.with(|s|s.borrow_mut().active=None);return Ok(result);} }
    else if matches!(status,Some(ReceiptState::Running)){continue;}
    return fail(id,"worker state mismatch".into()).await;
   }
   Ok(_)=>return fail(id,"worker progress mismatch".into()).await,
   Err(e)=>return fail(id,e).await,
  }
 }
}
#[ic_cdk::update]
fn inference_step(job_id:u64,expected_stage:u64)->Result<StepResult,String> {
 if ic_cdk::api::msg_caller()!=ic_cdk::api::canister_self(){return Err("self only".into());}
 let (request,q,graph_id)=PAID.with(|s|->Result<_,String>{let s=s.borrow();let j=s.active.as_ref().ok_or("missing job")?;
  if j.id!=job_id || j.stage!=expected_stage || !j.in_flight || j.workers.len()>=j.quote.max_steps as usize{return Err("job progress mismatch".into());}
  Ok((if expected_stage==0 {Some(j.request.clone())}else{None},j.quote.clone(),j.graph_id))})?;
 #[cfg(feature="paid-update-diagnostics")]
 if let Some((stage,trap))=FAULT.with(|f|*f.borrow()) {if stage==expected_stage {if trap{ic_cdk::trap("diagnostic worker trap");}return Err("diagnostic worker reject".into());}}
 let p=if let Some(r)=request {update_inference::select_bank(q.prefix_tokens as usize)?;update_inference::start(r.token_ids[q.prefix_tokens as usize..].to_vec(),r.options)?}else{update_inference::continue_graph(graph_id.ok_or("missing graph")?,expected_stage)?};
 let stage=p.stage;let done=p.done;
 let result=PAID.with(|s|->Result<Option<InferenceResult>,String>{let mut s=s.borrow_mut();let j=s.active.as_mut().ok_or("missing active job")?;if j.id!=job_id || j.stage!=expected_stage{return Err("stale job".into());}
  j.graph_id=Some(p.id);j.stage=stage;j.workers.push(WorkerMetric{stage,instructions:p.instructions,heap_pages:p.heap_pages});
  if done {Ok(Some(InferenceResult{job_id,decision:decision(p.decision.clone().ok_or("missing decision")?),paid_cycles:j.quote.fee,quote_version:j.quote.version,prefix_tokens:j.quote.prefix_tokens,suffix_tokens:j.quote.suffix_tokens,workers:j.workers.clone()}))}else{Ok(None)}})?;
 if let Some(r)=result {DEBUG.with(|d|*d.borrow_mut()=Some(DebugResult{job_id,hidden_hashes:p.hidden_hashes,state_hashes:p.state_hashes,final_hidden:p.final_hidden}));
  PAID.with(|s|{let mut s=s.borrow_mut();if let Some(row)=s.receipts.iter_mut().find(|r|r.job_id==job_id){row.state=ReceiptState::Completed(r);}});update_inference::release();}
 // Counter zero starts at the IC message entry, before CDK argument decoding.
 // Do not subtract a checkpoint taken inside this function: that omits the
 // wrapper/prologue. This remains a checkpoint; the metric update and reply
 // epilogue below still need a separate whole-message accounting bound.
 let measured=ic_cdk::api::performance_counter(0);
 PAID.with(|s|{let mut s=s.borrow_mut();if let Some(j)=s.active.as_mut(){if let Some(m)=j.workers.last_mut(){m.instructions=measured;}}
  if let Some(r)=s.receipts.iter_mut().find(|r|r.job_id==job_id){if let ReceiptState::Completed(v)=&mut r.state {if let Some(m)=v.workers.last_mut(){m.instructions=measured;}}}});
 Ok(StepResult{job_id,stage,done})
}
async fn fail(id:u64,reason:String)->Result<InferenceResult,InferError> {
 update_inference::release();PAID.with(|s|{let mut s=s.borrow_mut();s.active=None;if let Some(r)=s.receipts.iter_mut().find(|r|r.job_id==id){r.state=ReceiptState::Failed(reason.clone());r.refund=Refund::Pending;}});
 let refund=refund_job(id).await;Err(InferError::Failed{job_id:id,reason,refund})
}
#[derive(CandidType)]struct Deposit {canister_id:Principal}
async fn refund_job(id:u64)->Refund {
 let work=PAID.with(|s|{let mut s=s.borrow_mut();let r=s.receipts.iter_mut().find(|r|r.job_id==id)?;if r.refund!=Refund::Pending{return None;}r.refund=Refund::InFlight;Some((r.caller,r.quote.fee))});
 let Some((caller,fee))=work else{return PAID.with(|s|s.borrow().receipts.iter().find(|r|r.job_id==id).map(|r|r.refund.clone()).unwrap_or(Refund::None));};
 #[cfg(feature="paid-update-diagnostics")]
 if REFUND_FAULT.with(|f|*f.borrow()){PAID.with(|s|{if let Some(r)=s.borrow_mut().receipts.iter_mut().find(|r|r.job_id==id){r.refund=Refund::Pending;}});return Refund::Pending;}
 let response=Call::unbounded_wait(Principal::management_canister(),"deposit_cycles").with_arg(Deposit{canister_id:caller}).with_cycles(fee).await;
 let state=match response{Ok(_)=>Refund::Done,Err(_)=>Refund::Pending};
 PAID.with(|s|{if let Some(r)=s.borrow_mut().receipts.iter_mut().find(|r|r.job_id==id){r.refund=state.clone();}});state
}
#[ic_cdk::update]
async fn retry_inference_refund(request_id:String)->Result<Refund,String>{let caller=ic_cdk::api::msg_caller();let id=PAID.with(|s|s.borrow().receipts.iter().find(|r|r.caller==caller && r.request_id==request_id).map(|r|r.job_id)).ok_or("missing receipt")?;Ok(refund_job(id).await)}
#[ic_cdk::query]
fn inference_status(request_id:String)->Option<Receipt>{let caller=ic_cdk::api::msg_caller();PAID.with(|s|s.borrow().receipts.iter().find(|r|r.caller==caller && r.request_id==request_id).cloned())}
#[ic_cdk::query]
fn paid_debug()->Option<DebugResult>{owner();DEBUG.with(|d|d.borrow().clone())}
#[ic_cdk::query]
fn paid_config()->Config {PAID.with(|s|s.borrow().config.clone())}
pub(super) fn admin_guard(){PAID.with(|s|assert!(s.borrow().active.is_none(),"paid inference active"));}
pub(super) fn before_upgrade(){admin_guard();PAID.with(|s|assert!(!s.borrow().receipts.iter().any(|r|r.refund==Refund::InFlight),"refund in flight"));}
pub(super) fn metadata()->Vec<u8>{PAID.with(|s|serde_json::to_vec(&*s.borrow()).expect("paid metadata"))}
pub(super) fn restore_metadata(bytes:&[u8]){if !bytes.is_empty(){let s:PaidStore=serde_json::from_slice(bytes).expect("paid metadata");assert!(s.active.is_none() && s.receipts.len()<=RECEIPTS);PAID.with(|p|*p.borrow_mut()=s);}}
#[cfg(feature="paid-update-diagnostics")]
#[ic_cdk::update]
fn paid_fault(stage:Option<u64>,trap:bool,refund_fail:bool){owner();FAULT.with(|f|*f.borrow_mut()=stage.map(|s|(s,trap)));REFUND_FAULT.with(|f|*f.borrow_mut()=refund_fail);}

// Diagnostic-only route exercises stale internal requests while a job is active.
#[cfg(feature="paid-update-diagnostics")]
#[ic_cdk::update]
async fn paid_probe_step(job_id:u64,stage:u64)->Result<StepResult,String>{
 if !STORE.with(|s|s.borrow().owner==Some(ic_cdk::api::msg_caller())){return Err("owner only".into());}
 Call::unbounded_wait(ic_cdk::api::canister_self(),"inference_step").with_args(&(job_id,stage)).await.map_err(|e|e.to_string())?.candid::<Result<StepResult,String>>().map_err(|e|e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request_with_suffix(n:usize)->InferRequest {
        let mut token_ids=PREFIX27.to_vec();token_ids.extend(vec![1;n]);
        InferRequest{model:"0".repeat(64),version:1,token_ids,options:vec!["no".into(),"yes".into()]}
    }
    #[test]
    fn common_prefix_accepts_84_tokens_but_rejects_85_and_empty_suffix() {
        assert_eq!(suffix_tokens(&request_with_suffix(57)).unwrap(),57);
        assert!(matches!(suffix_tokens(&request_with_suffix(58)),Err(InferError::Invalid(_))));
        assert!(matches!(suffix_tokens(&request_with_suffix(0)),Err(InferError::Invalid(_))));
    }
    #[test]
    fn former_voting_prefix_is_part_of_the_paid_suffix() {
        let mut r=request_with_suffix(0);
        r.token_ids.extend([27756,15209,70173,7383,4203,494,220,16,1834,310,220]);
        r.token_ids.push(1);
        assert_eq!(suffix_tokens(&r).unwrap(),12);
        r.token_ids[0]=1;
        assert!(matches!(suffix_tokens(&r),Err(InferError::Invalid(_))));
    }
    #[test]
    fn pause_during_inference_preserves_job_quote_and_blocks_reopening() {
        let quote=Quote{version:2,fee:268_000_000_000,prefix_tokens:38,suffix_tokens:56,estimated_steps:5,max_steps:6};
        PAID.with(|p| {let mut s=p.borrow_mut();*s=PaidStore::default();
            s.config=Config{enabled:true,version:2,base_fee:100_000_000_000,fee_per_token:3_000_000_000,reserve_cycles:2_000_000_000_000};
            s.active=Some(Job{id:1,stage:16,graph_id:Some(1),request:InferRequest{model:"0".repeat(64),version:1,token_ids:vec![],options:vec![]},quote:quote.clone(),in_flight:true,workers:vec![]});
        });
        let mut config=PAID.with(|p|p.borrow().config.clone());config.enabled=false;config.version=3;
        assert!(apply_config(config.clone()).is_ok());
        PAID.with(|p| {let s=p.borrow();assert!(!s.config.enabled);let j=s.active.as_ref().unwrap();
            assert_eq!(j.quote.version,quote.version);assert_eq!(j.quote.fee,quote.fee);assert_eq!(j.stage,16);assert!(j.in_flight);
        });
        config.enabled=true;config.version=4;
        assert_eq!(apply_config(config.clone()),Err("paid inference active".into()));
        assert_eq!(PAID.with(|p|p.borrow().config.version),3);
        PAID.with(|p|p.borrow_mut().active=None);
        assert!(apply_config(config).is_ok());
    }
    fn failed_receipt(id:u64, refund:Refund)->Receipt {
        Receipt {caller:Principal::from_slice(&[1,2,3]),request_id:format!("failed-{id}"),input_hash:"a".repeat(64),job_id:id,
            quote:Quote{version:2,fee:268_000_000_000,prefix_tokens:38,suffix_tokens:56,estimated_steps:5,max_steps:6},
            time:1,state:ReceiptState::Failed("worker rejected".into()),refund}
    }
    #[test]
    fn failed_receipts_and_refunds_roundtrip_upgrade_metadata() {
        PAID.with(|p| {let mut s=p.borrow_mut();*s=PaidStore::default();s.next_id=2;s.receipts=vec![failed_receipt(1,Refund::Done),failed_receipt(2,Refund::Pending)];});
        let saved=metadata();
        PAID.with(|p|*p.borrow_mut()=PaidStore::default());
        restore_metadata(&saved);
        assert_eq!(metadata(),saved);
        PAID.with(|p| {let s=p.borrow();assert_eq!(s.next_id,2);assert_eq!(s.receipts[0].refund,Refund::Done);assert_eq!(s.receipts[1].refund,Refund::Pending);});
    }
    #[test]
    #[should_panic(expected="refund in flight")]
    fn upgrade_refuses_unresolved_transfer() {
        PAID.with(|p| {let mut s=p.borrow_mut();*s=PaidStore::default();s.receipts=vec![failed_receipt(1,Refund::InFlight)];});
        before_upgrade();
    }
}

// Diagnostic fixture for upgrade guards and stable receipts, without loading weights.
#[cfg(feature="paid-update-diagnostics")]
#[ic_cdk::update]
fn paid_upgrade_probe(active:bool,refund_state:u32){
 let caller=ic_cdk::api::msg_caller();assert!(STORE.with(|s|s.borrow().owner==Some(caller)),"owner only");assert!(refund_state<=3);
 let quote=Quote{version:1,fee:1,prefix_tokens:27,suffix_tokens:1,estimated_steps:1,max_steps:6};
 PAID.with(|p|{let mut s=p.borrow_mut();s.active=if active{Some(Job{id:u64::MAX,stage:0,graph_id:None,request:InferRequest{model:"0".repeat(64),version:1,token_ids:PREFIX27.to_vec(),options:vec!["no".into(),"yes".into()]},quote:quote.clone(),in_flight:false,workers:vec![]})}else{None};
 s.receipts=if refund_state==0{vec![]}else{vec![Receipt{caller,request_id:"diagnostic-upgrade".into(),input_hash:"0".repeat(64),job_id:u64::MAX,quote,time:ic_cdk::api::time(),state:ReceiptState::Failed("diagnostic receipt".into()),refund:match refund_state{1=>Refund::Pending,2=>Refund::InFlight,_=>Refund::Done}}]};});
}
