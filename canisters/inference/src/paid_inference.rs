//! Caller-funded inference. One active graph; each heavy batch is a new self-call.
use super::*;
use super::paid_types::*;
use ic_cdk::call::Call;
use serde::{Serialize,Deserialize};
#[path="receipt_archive.rs"]
mod receipt_archive;
const RECEIPTS:usize=receipt_archive::CAPACITY;
const TTL:u64=86_400_000_000_000;
const MAX_STEPS:u32=8;
// Replay must survive switching back to a build with a smaller execution limit.
const REPLAY_INPUT_LIMIT:usize=1024;
#[cfg(feature="experimental-update-token-chunks")]
pub(super) const INPUT_LIMIT:usize=imajev_runtime::MAX_SEQUENCE_TOKENS;
#[cfg(not(feature="experimental-update-token-chunks"))]
pub(super) const INPUT_LIMIT:usize=COMMON_PREFIX.len()+update_inference::UNCHUNKED_SUFFIX_LIMIT;
pub(super) const COMMON_PREFIX:[u32;5]=[248045,846,198,1349,25];
#[derive(Serialize,Deserialize)]
struct PaidStore {config:Config,next_id:u64,receipts:Vec<Receipt>,active:Option<Job>,#[serde(default)] archive:receipt_archive::Archive}
#[derive(Serialize,Deserialize)]
struct Job {id:u64,stage:u64,graph_id:Option<u64>,request:InferRequest,quote:Quote,in_flight:bool,workers:Vec<WorkerMetric>}
impl Default for PaidStore {fn default()->Self {Self{config:Config{fee_per_token:0,base_fee:0,reserve_cycles:0},next_id:0,receipts:vec![],active:None,archive:Default::default()}}}
thread_local! {static PAID:RefCell<PaidStore>=RefCell::new(PaidStore::default());static DEBUG:RefCell<Option<DebugResult>>=RefCell::new(None);}
#[cfg(feature="paid-update-diagnostics")]
thread_local! {static FAULT:RefCell<Option<(u64,bool)>>=RefCell::new(None);static REFUND_FAULT:RefCell<bool>=RefCell::new(false);}
fn decision(d:ChoiceResult)->Decision {Decision{value:d.value,probabilities:d.probabilities,unknown_probability:d.unknown_probability,abstained:d.abstained,raw_logits:d.raw_logits,instructions:d.instructions,calibration_version:d.calibration_version}}
fn suffix_tokens(r:&InferRequest)->Result<usize,InferError> {
 if r.version!=1 || r.model.len()!=64 || r.token_ids.len()>INPUT_LIMIT || r.token_ids.is_empty() || r.token_ids.iter().any(|i|*i>=248320) {return Err(InferError::Invalid("input bounds/version".into()));}
 if !r.token_ids.starts_with(&COMMON_PREFIX){return Err(InferError::Invalid("common prefix mismatch".into()));}
 let n=r.token_ids.len()-COMMON_PREFIX.len();
 if !(1..=INPUT_LIMIT-COMMON_PREFIX.len()).contains(&n){return Err(InferError::Invalid("suffix bounds".into()));}
 Ok(n)
}
// The quote is advisory: admission recalculates the current fee before accepting cycles.
fn fee_for_suffix(config:&Config,tokens:usize)->Result<u128,InferError>{
 let fee=config.fee_per_token.checked_mul(tokens as u128).and_then(|v|v.checked_add(config.base_fee))
  .ok_or_else(||InferError::Invalid("fee overflow".into()))?;
 if fee==0{return Err(InferError::NotReady);}
 Ok(fee)
}
struct Admission {caller:Principal,request_id:String,request:InferRequest,input_hash:String,quote:Quote,time:u64}
fn accept_job(a:Admission,available:u128,liquid_balance:u128,graph_busy:bool,accept:impl FnOnce(u128)->u128)->Result<u64,InferError>{
 PAID.with(|p|{let mut s=p.borrow_mut();
  if s.active.is_some() || graph_busy{return Err(InferError::Busy);}
  if s.receipts.len()+s.archive.len()>=RECEIPTS{return Err(InferError::ReceiptCapacity);}
  if available<a.quote.fee{return Err(InferError::InsufficientCycles{required:a.quote.fee});}
  if liquid_balance<s.config.reserve_cycles.saturating_add(a.quote.fee){return Err(InferError::NotReady);}
  let id=s.next_id.checked_add(1).ok_or(InferError::ReceiptCapacity)?;
  let accepted=accept(a.quote.fee);assert_eq!(accepted,a.quote.fee,"cycles acceptance");s.next_id=id;
  s.receipts.push(Receipt{caller:a.caller,request_id:a.request_id,input_hash:a.input_hash,job_id:id,quote:a.quote.clone(),time:a.time,state:ReceiptState::Running,refund:Refund::None});
  s.active=Some(Job{id,stage:0,graph_id:None,request:a.request,quote:a.quote,in_flight:false,workers:vec![]});
  Ok(id)
 })
}
fn plan(r:&InferRequest)->Result<Quote,InferError> {
 let n=suffix_tokens(r)?;
 imajev_runtime::decide_candidates(&r.options,&vec![0.;r.options.len().min(7)+1],1.3051569717552742).map_err(InferError::Invalid)?;
 let valid=STORE.with(|s|s.borrow().manifest.as_ref().is_some_and(|m|m.model==r.model));
 if !valid {return Err(InferError::Invalid("model mismatch".into()));}
 let p=COMMON_PREFIX.len();
 if !update_inference::bank_ready(p){return Err(InferError::NotReady);}
 if !STORE.with(|s|s.borrow().ready && s.borrow().weight_cache.names().len()==721) {return Err(InferError::NotReady);}
 PAID.with(|s|{let s=s.borrow();let fee=fee_for_suffix(&s.config,n)?;
  Ok(Quote{fee,prefix_tokens:p as u32,suffix_tokens:n as u32,estimated_steps:((n as u64*2_600_000_000).div_ceil(worker_budget()) as u32),max_steps:if reference_enabled() {256}else if n>507 {128}else if n>229 {64}else if n>89 {32}else{MAX_STEPS}})})
}
fn reference_enabled()->bool {
 #[cfg(feature="paid-update-diagnostics")] {return imajev_runtime::attention_reference_enabled();}
 #[cfg(not(feature="paid-update-diagnostics"))] {false}
}
pub(super) fn worker_budget()->u64 {if reference_enabled(){20_000_000_000}else{update_inference::WORKER_BUDGET}}
#[cfg(feature="paid-update-diagnostics")]
#[ic_cdk::update(name = "setPaidInferenceReference")]
fn paid_reference(enabled:bool){owner();imajev_runtime::set_attention_reference(enabled);}
#[ic_cdk::query(name = "getInferenceQuote")]
fn quote(request:InferRequest)->Result<Quote,InferError>{plan(&request)}
#[ic_cdk::update(name = "configurePaidInference")]
fn configure_paid(config:Config)->Result<(),String>{owner_auth();apply_config(config)}
fn apply_config(config:Config)->Result<(),String>{if config.base_fee>1_000_000_000_000_000 || config.fee_per_token>1_000_000_000_000 || config.reserve_cycles>1_000_000_000_000_000 || config.base_fee==0{return Err("config bounds".into());}
 PAID.with(|s|{let mut s=s.borrow_mut();s.config=config;Ok(())})}
fn input_hash(request:&InferRequest)->String {let b=candid::encode_one(request).expect("validated request encoding");Sha256::digest(b).iter().map(|b|format!("{b:02x}")).collect()}
fn validate_replay_size(request:&InferRequest)->Result<(),InferError> {
 if request.token_ids.len()>REPLAY_INPUT_LIMIT || request.options.len()>7 || request.options.iter().any(|s|s.len()>128){return Err(InferError::Invalid("input size".into()));}
 Ok(())
}
fn replay_receipt(r:Receipt,hash:&str)->Result<InferenceResult,InferError> {
 if r.input_hash!=hash{return Err(InferError::IdConflict);}
 match r.state {ReceiptState::Completed(v)=>Ok(v),ReceiptState::Running=>Err(InferError::InProgress{job_id:r.job_id}),ReceiptState::Failed(reason)=>Err(InferError::Failed{job_id:r.job_id,reason,refund:r.refund})}
}
fn find_receipt(s:&PaidStore,caller:Principal,request_id:&str)->Option<Receipt> {
 s.receipts.iter().find(|r|r.caller==caller && r.request_id==request_id).cloned().or_else(||s.archive.find(caller,request_id))
}
fn receipt_by_job(s:&PaidStore,id:u64)->Option<Receipt> {s.receipts.iter().find(|r|r.job_id==id).cloned().or_else(||s.archive.by_job(id))}
fn edit_receipt(s:&mut PaidStore,id:u64,edit:impl FnOnce(&mut Receipt))->Option<()> {
 if let Some(r)=s.receipts.iter_mut().find(|r|r.job_id==id){edit(r);return Some(());}
 let mut r=s.archive.by_job(id)?;edit(&mut r);s.archive.store(&r);Some(())
}
fn archive_receipts(s:&mut PaidStore) {
 for r in std::mem::take(&mut s.receipts) {
  if matches!(r.state,ReceiptState::Running) || r.refund==Refund::InFlight || s.active.as_ref().is_some_and(|j|j.id==r.job_id) {s.receipts.push(r);}
  else {s.archive.store(&r);}
 }
}
fn prune_at(s:&mut PaidStore,now:u64) {
 let active=s.active.as_ref().map(|j|j.id);
 s.receipts.retain(|r|active==Some(r.job_id) || matches!(r.state,ReceiptState::Running) || !matches!(r.refund,Refund::None|Refund::Done) || now.saturating_sub(r.time)<TTL);
 archive_receipts(s);s.archive.expire(now,TTL);
}
fn prune(s:&mut PaidStore){prune_at(s,ic_cdk::api::time());}
#[ic_cdk::update(name = "runPaidInference")]
async fn infer(request:InferRequest,request_id:String)->Result<InferenceResult,InferError> {
 let caller=ic_cdk::api::msg_caller();
 if caller==Principal::anonymous() || caller==ic_cdk::api::canister_self() || request_id.is_empty() || request_id.len()>64 {return Err(InferError::Invalid("caller/request ID".into()));}
 // Receipts remain bound to caller, request ID and the full input hash.
 validate_replay_size(&request)?;
 let hash=input_hash(&request);
 let duplicate=PAID.with(|s|{let mut s=s.borrow_mut();prune(&mut s);find_receipt(&s,caller,&request_id)});
 if let Some(r)=duplicate {return replay_receipt(r,&hash);}
 let q=plan(&request)?;
 let id=accept_job(Admission{caller,request_id,request,input_hash:hash,quote:q.clone(),time:ic_cdk::api::time()},
  ic_cdk::api::msg_cycles_available(),ic_cdk::api::canister_liquid_cycle_balance(),update_inference::busy(),
  ic_cdk::api::msg_cycles_accept)?;
 loop {
  let stage=PAID.with(|s|{let mut s=s.borrow_mut();let j=s.active.as_mut().expect("active job");assert_eq!(j.id,id);j.in_flight=true;j.stage});
  let response=Call::unbounded_wait(ic_cdk::api::canister_self(),"runPaidInferenceWorker").with_args(&(id,stage)).await;
  let decoded=response.map_err(|e|format!("worker call: {e}")).and_then(|v|v.candid::<Result<StepResult,String>>().map_err(|e|format!("worker decode: {e}")).and_then(|v|v));
  match decoded {
   Ok(v) if v.job_id==id && v.stage>stage && v.stage<=update_inference::progress_limit(q.suffix_tokens as usize) => {
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
#[ic_cdk::update(name = "runPaidInferenceWorker")]
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
  if done {Ok(Some(InferenceResult{job_id,decision:decision(p.decision.clone().ok_or("missing decision")?),paid_cycles:j.quote.fee,prefix_tokens:j.quote.prefix_tokens,suffix_tokens:j.quote.suffix_tokens,workers:j.workers.clone()}))}else{Ok(None)}})?;
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
async fn fail(id:u64,mut reason:String)->Result<InferenceResult,InferError> {
 // Platform reject messages must not make a retained receipt exceed its slot.
 if reason.len()>8192 {let mut end=8192;while !reason.is_char_boundary(end){end-=1;}reason.truncate(end);reason.push_str(" [truncated]");}
 update_inference::release();PAID.with(|s|{let mut s=s.borrow_mut();s.active=None;if let Some(r)=s.receipts.iter_mut().find(|r|r.job_id==id){r.state=ReceiptState::Failed(reason.clone());r.refund=Refund::Pending;}});
 let refund=refund_job(id).await;Err(InferError::Failed{job_id:id,reason,refund})
}
#[derive(CandidType)]struct Deposit {canister_id:Principal}
async fn refund_job(id:u64)->Refund {
 let work=PAID.with(|s|{let mut s=s.borrow_mut();let r=receipt_by_job(&s,id)?;if r.refund!=Refund::Pending{return None;}edit_receipt(&mut s,id,|r|r.refund=Refund::InFlight);Some((r.caller,r.quote.fee))});
 let Some((caller,fee))=work else{return PAID.with(|s|receipt_by_job(&s.borrow(),id).map(|r|r.refund).unwrap_or(Refund::None));};
 #[cfg(feature="paid-update-diagnostics")]
 if REFUND_FAULT.with(|f|*f.borrow()){PAID.with(|s|{edit_receipt(&mut s.borrow_mut(),id,|r|r.refund=Refund::Pending);});return Refund::Pending;}
 let response=Call::unbounded_wait(Principal::management_canister(),"deposit_cycles").with_arg(Deposit{canister_id:caller}).with_cycles(fee).await;
 let state=match response{Ok(_)=>Refund::Done,Err(_)=>Refund::Pending};
 PAID.with(|s|{edit_receipt(&mut s.borrow_mut(),id,|r|r.refund=state.clone());});state
}
#[ic_cdk::update(name = "retryInferenceRefund")]
async fn retry_inference_refund(request_id:String)->Result<Refund,String>{let caller=ic_cdk::api::msg_caller();let id=PAID.with(|s|find_receipt(&s.borrow(),caller,&request_id).map(|r|r.job_id)).ok_or("missing receipt")?;Ok(refund_job(id).await)}
#[ic_cdk::query(name = "getInferenceReceipt")]
fn inference_status(request_id:String)->Option<Receipt>{let caller=ic_cdk::api::msg_caller();PAID.with(|s|find_receipt(&s.borrow(),caller,&request_id))}
#[cfg(feature="paid-update-diagnostics")]
#[ic_cdk::query(name = "getPaidInferenceDebug")]
fn paid_debug()->Option<DebugResult>{owner();DEBUG.with(|d|d.borrow().clone())}
#[ic_cdk::query(name = "getPaidInferenceConfig")]
fn paid_config()->Config {PAID.with(|s|s.borrow().config.clone())}
pub(super) fn admin_guard(){PAID.with(|s|assert!(s.borrow().active.is_none(),"paid inference active"));}
pub(super) fn before_upgrade(){admin_guard();PAID.with(|s|{let mut s=s.borrow_mut();assert!(!s.receipts.iter().any(|r|r.refund==Refund::InFlight) && !s.archive.in_flight(),"refund in flight");archive_receipts(&mut s);});}
pub(super) fn metadata()->Vec<u8>{PAID.with(|s|serde_json::to_vec(&*s.borrow()).expect("paid metadata"))}
pub(super) fn restore_metadata(bytes:&[u8]){if !bytes.is_empty(){let mut s:PaidStore=serde_json::from_slice(bytes).expect("paid metadata");s.archive.restore();assert!(s.active.is_none() && s.receipts.len()+s.archive.len()<=RECEIPTS);PAID.with(|p|*p.borrow_mut()=s);}}
#[cfg(feature="paid-update-diagnostics")]
#[ic_cdk::update(name = "setPaidInferenceFault")]
fn paid_fault(stage:Option<u64>,trap:bool,refund_fail:bool){owner();FAULT.with(|f|*f.borrow_mut()=stage.map(|s|(s,trap)));REFUND_FAULT.with(|f|*f.borrow_mut()=refund_fail);}

// Diagnostic-only route exercises stale internal requests while a job is active.
#[cfg(feature="paid-update-diagnostics")]
#[ic_cdk::update(name = "probePaidInferenceWorker")]
async fn paid_probe_step(job_id:u64,stage:u64)->Result<StepResult,String>{
 if !STORE.with(|s|s.borrow().owner==Some(ic_cdk::api::msg_caller())){return Err("owner only".into());}
 Call::unbounded_wait(ic_cdk::api::canister_self(),"runPaidInferenceWorker").with_args(&(job_id,stage)).await.map_err(|e|e.to_string())?.candid::<Result<StepResult,String>>().map_err(|e|e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request_with_suffix(n:usize)->InferRequest {
        let mut token_ids=COMMON_PREFIX.to_vec();token_ids.extend(vec![1;n]);
        InferRequest{model:"0".repeat(64),version:1,token_ids,options:vec!["no".into(),"yes".into()]}
    }
    #[test]
    fn common_prefix_accepts_feature_limit_and_rejects_overflow_and_empty_suffix() {
        assert_eq!(suffix_tokens(&request_with_suffix(INPUT_LIMIT-5)).unwrap(),INPUT_LIMIT-5);
        assert!(matches!(suffix_tokens(&request_with_suffix(INPUT_LIMIT-4)),Err(InferError::Invalid(_))));
        assert!(matches!(suffix_tokens(&request_with_suffix(0)),Err(InferError::Invalid(_))));
    }
    #[test]
    #[cfg(not(feature="experimental-update-token-chunks"))]
    fn unchunked_admission_matches_scheduler_limit() {
        let n=update_inference::UNCHUNKED_SUFFIX_LIMIT;
        assert_eq!(INPUT_LIMIT,COMMON_PREFIX.len()+n);
        assert_eq!(suffix_tokens(&request_with_suffix(n)).unwrap(),n);
        assert!(matches!(suffix_tokens(&request_with_suffix(n+1)),Err(InferError::Invalid(_))));
    }
    #[test]
    fn replay_bounds_do_not_expand_new_inference_limit() {
        let request=request_with_suffix(REPLAY_INPUT_LIMIT-5);
        assert!(validate_replay_size(&request).is_ok());
        #[cfg(not(feature="experimental-update-token-chunks"))]
        assert!(matches!(suffix_tokens(&request),Err(InferError::Invalid(_))));
        #[cfg(feature="experimental-update-token-chunks")]
        assert_eq!(suffix_tokens(&request).unwrap(),REPLAY_INPUT_LIMIT-5);
        assert!(matches!(validate_replay_size(&request_with_suffix(REPLAY_INPUT_LIMIT+1-5)),Err(InferError::Invalid(_))));
        let mut oversized=request.clone();oversized.options=vec!["no".into();8];
        assert!(matches!(validate_replay_size(&oversized),Err(InferError::Invalid(_))));
        oversized.options=vec!["x".repeat(129)];
        assert!(matches!(validate_replay_size(&oversized),Err(InferError::Invalid(_))));
    }
    #[test]
    fn restored_512_token_receipts_replay_results_refunds_and_conflicts() {
        let mut request=request_with_suffix(0);
        request.token_ids=vec![248045,846,198,56555,279,2420,5721,321,4087,279,3296,1608,279,10661,12521,13,3301,1132,279,3074,2904,1970,13,198,1349,25,328];
        request.token_ids.extend(vec![1;485]);
        let hash=input_hash(&request);
        let result=InferenceResult{job_id:1,decision:Decision{value:Some("yes".into()),probabilities:vec![0.2,0.7],
            unknown_probability:0.1,abstained:false,raw_logits:vec![1.,2.,0.],instructions:100,calibration_version:"test".into()},
            paid_cycles:1_555_000_000_000,prefix_tokens:27,suffix_tokens:485,workers:vec![]};
        let mut completed=failed_receipt(1,Refund::None);
        completed.input_hash=hash.clone();completed.quote.suffix_tokens=485;completed.quote.prefix_tokens=27;
        completed.quote.max_steps=64;completed.state=ReceiptState::Completed(result.clone());
        let mut failed=completed.clone();failed.job_id=2;failed.request_id="failed-2".into();
        failed.state=ReceiptState::Failed("worker rejected".into());failed.refund=Refund::Pending;
        PAID.with(|p|{let mut s=p.borrow_mut();*s=PaidStore::default();s.next_id=2;s.receipts=vec![completed,failed];});
        let saved=metadata();
        PAID.with(|p|*p.borrow_mut()=PaidStore::default());
        restore_metadata(&saved);
        validate_replay_size(&request).unwrap();
        let receipts=PAID.with(|p|p.borrow().receipts.clone());
        let replayed=replay_receipt(receipts[0].clone(),&hash).unwrap();
        assert_eq!(candid::encode_one(replayed).unwrap(),candid::encode_one(result).unwrap());
        assert!(matches!(replay_receipt(receipts[1].clone(),&hash),Err(InferError::Failed{job_id:2,refund:Refund::Pending,..})));
        let mut changed=request;changed.token_ids[27]=2;
        assert!(matches!(replay_receipt(receipts[0].clone(),&input_hash(&changed)),Err(InferError::IdConflict)));
        assert_eq!(metadata(),saved);
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
    fn fee_changes_during_inference_preserve_the_admitted_job_quote() {
        configure_test_tariff(10,3);
        let fee=test_admission().quote.fee;
        accept_job(test_admission(),fee,u128::MAX,false,|n|n).unwrap();
        apply_config(Config{base_fee:20,fee_per_token:4,reserve_cycles:0}).unwrap();
        PAID.with(|p| {let s=p.borrow();assert_eq!(s.active.as_ref().unwrap().quote.fee,fee);assert_eq!(s.receipts[0].quote.fee,fee);});
        assert!(matches!(accept_job(test_admission(),u128::MAX,u128::MAX,false,|_|panic!("busy payment")),Err(InferError::Busy)));
    }
    fn configure_test_tariff(base_fee:u128,fee_per_token:u128) {
        PAID.with(|p|*p.borrow_mut()=PaidStore::default());
        apply_config(Config{base_fee,fee_per_token,reserve_cycles:0}).unwrap();
    }
    fn test_admission() -> Admission {
        let request=request_with_suffix(57);
        let fee=PAID.with(|p|fee_for_suffix(&p.borrow().config,57)).unwrap();
        Admission{caller:Principal::from_slice(&[1,2,3]),request_id:"payment-test".into(),input_hash:input_hash(&request),request,
            quote:Quote{fee,prefix_tokens:5,suffix_tokens:57,estimated_steps:5,max_steps:8},time:1}
    }
    #[test]
    fn price_increase_rejects_old_attachment_without_accepting_or_creating_job() {
        configure_test_tariff(10,3);
        let old_fee=test_admission().quote.fee;
        apply_config(Config{base_fee:20,fee_per_token:3,reserve_cycles:0}).unwrap();
        let current=test_admission();let required=current.quote.fee;
        let result=accept_job(current,old_fee,u128::MAX,false,|_|panic!("must not accept cycles"));
        assert!(matches!(result,Err(InferError::InsufficientCycles{required:n}) if n==required));
        PAID.with(|p|{let s=p.borrow();assert_eq!(s.next_id,0);assert!(s.active.is_none());assert!(s.receipts.is_empty());});
        // A fresh request with sufficient attachment can be admitted after the rejection.
        assert_eq!(accept_job(test_admission(),required,u128::MAX,false,|fee|fee).unwrap(),1);
    }
    #[test]
    fn price_decrease_accepts_only_current_fee_and_later_changes_keep_job_fee() {
        configure_test_tariff(20,3);
        let old_fee=test_admission().quote.fee;
        apply_config(Config{base_fee:10,fee_per_token:3,reserve_cycles:0}).unwrap();
        let current_fee=test_admission().quote.fee;
        assert!(current_fee<old_fee);
        let mut accepted=0;
        accept_job(test_admission(),old_fee,u128::MAX,false,|fee|{accepted=fee;fee}).unwrap();
        assert_eq!(accepted,current_fee);
        apply_config(Config{base_fee:99,fee_per_token:9,reserve_cycles:0}).unwrap();
        PAID.with(|p|{let s=p.borrow();assert_eq!(s.active.as_ref().unwrap().quote.fee,current_fee);assert_eq!(s.receipts[0].quote.fee,current_fee);});
    }
    #[test]
    fn busy_capacity_and_reserve_reject_before_payment() {
        configure_test_tariff(10,3);
        assert!(matches!(accept_job(test_admission(),u128::MAX,u128::MAX,true,|_|panic!("busy payment")),Err(InferError::Busy)));
        assert!(matches!(accept_job(test_admission(),u128::MAX,0,false,|_|panic!("reserve payment")),Err(InferError::NotReady)));
        PAID.with(|p|p.borrow_mut().receipts=(0..RECEIPTS).map(|id|failed_receipt(id as u64,Refund::Pending)).collect());
        assert!(matches!(accept_job(test_admission(),u128::MAX,u128::MAX,false,|_|panic!("capacity payment")),Err(InferError::ReceiptCapacity)));
        PAID.with(|p|assert!(p.borrow().active.is_none()));
    }
    #[test]
    fn terminal_receipt_stays_hot_until_the_inference_callback_finishes() {
        configure_test_tariff(10,3);let id=accept_job(test_admission(),u128::MAX,u128::MAX,false,|n|n).unwrap();
        PAID.with(|p|{let mut s=p.borrow_mut();s.receipts[0].state=ReceiptState::Failed("callback pending".into());
            prune_at(&mut s,1);assert_eq!(s.receipts.len(),1);assert_eq!(s.archive.len(),0);
            assert_eq!(s.receipts[0].job_id,id);s.active=None;prune_at(&mut s,1);assert!(s.receipts.is_empty());assert_eq!(s.archive.len(),1);});
    }
    #[test]
    fn active_terminal_receipt_is_not_pruned_after_ttl() {
        configure_test_tariff(10,3);accept_job(test_admission(),u128::MAX,u128::MAX,false,|n|n).unwrap();
        PAID.with(|p|{let mut s=p.borrow_mut();s.receipts[0].state=ReceiptState::Failed("callback pending".into());
            prune_at(&mut s,TTL+1);assert_eq!(s.receipts.len(),1);assert_eq!(s.archive.len(),0);
            s.active=None;prune_at(&mut s,TTL+1);assert!(s.receipts.is_empty());assert_eq!(s.archive.len(),0);});
    }
    #[test]
    fn stable_receipts_accept_beyond_128_and_preserve_replay_after_upgrade() {
        configure_test_tariff(10,3);
        let mut last=None;
        for i in 0..130 {
            let mut a=test_admission();a.request_id=format!("receipt-{i}");
            let caller=a.caller;let request_id=a.request_id.clone();let hash=a.input_hash.clone();
            let id=accept_job(a,u128::MAX,u128::MAX,false,|n|n).unwrap();
            PAID.with(|p|{let mut s=p.borrow_mut();let row=&mut s.receipts[0];
                row.state=ReceiptState::Completed(InferenceResult{job_id:id,decision:Decision{value:Some("yes".into()),probabilities:vec![0.9,0.05],unknown_probability:0.05,abstained:false,raw_logits:vec![1.,0.,0.],instructions:1,calibration_version:"test".into()},paid_cycles:row.quote.fee,prefix_tokens:5,suffix_tokens:57,workers:vec![]});s.active=None;prune_at(&mut s,1);});
            last=Some((caller,request_id,hash));
        }
        let saved=metadata();assert!(saved.len()<1024,"upgrade metadata includes only the archive descriptor");
        configure_test_tariff(10,3);restore_metadata(&saved);
        PAID.with(|p|{let s=p.borrow();assert_eq!(s.archive.len(),130);assert!(s.receipts.is_empty());
            let (caller,request_id,hash)=last.unwrap();let receipt=find_receipt(&s,caller,&request_id).unwrap();
            assert!(replay_receipt(receipt.clone(),&hash).is_ok());assert!(matches!(replay_receipt(receipt,"wrong hash"),Err(InferError::IdConflict)));});
    }
    #[test]
    fn archived_same_request_id_remains_scoped_to_each_caller_after_upgrade() {
        configure_test_tariff(10,3);
        let mut first=failed_receipt(1,Refund::Pending);
        first.request_id="shared-id".into();
        let mut second=first.clone();second.caller=Principal::from_slice(&[4,5,6]);second.job_id=2;second.refund=Refund::Done;
        PAID.with(|p|{let mut s=p.borrow_mut();s.next_id=2;s.archive.store(&first);s.archive.store(&second);});
        before_upgrade();let saved=metadata();configure_test_tariff(10,3);restore_metadata(&saved);
        PAID.with(|p|{let s=p.borrow();
            assert_eq!(find_receipt(&s,first.caller,"shared-id").unwrap().refund,Refund::Pending);
            assert_eq!(find_receipt(&s,second.caller,"shared-id").unwrap().refund,Refund::Done);
            assert!(find_receipt(&s,Principal::anonymous(),"shared-id").is_none());
        });
    }
    #[test]
    fn archived_pending_refunds_survive_expiry_and_transfer_guards() {
        configure_test_tariff(10,3);
        PAID.with(|p|{let mut s=p.borrow_mut();s.receipts=vec![failed_receipt(1,Refund::Pending),failed_receipt(2,Refund::Done)];prune_at(&mut s,1);
            assert_eq!(s.archive.len(),2);prune_at(&mut s,TTL+1);assert_eq!(s.archive.len(),1);
            edit_receipt(&mut s,1,|r|r.refund=Refund::InFlight).unwrap();assert!(s.archive.in_flight());
            edit_receipt(&mut s,1,|r|r.refund=Refund::Done).unwrap();assert!(!s.archive.in_flight());});
        before_upgrade();let saved=metadata();restore_metadata(&saved);
        PAID.with(|p|{let mut s=p.borrow_mut();assert_eq!(receipt_by_job(&s,1).unwrap().refund,Refund::Done);prune_at(&mut s,TTL+1);assert_eq!(s.archive.len(),0);
            s.archive.store(&failed_receipt(3,Refund::Pending));assert_eq!(s.archive.len(),1);});
    }
    #[test]
    #[should_panic(expected="refund in flight")]
    fn upgrade_refuses_archived_in_flight_refund() {
        configure_test_tariff(10,3);
        PAID.with(|p|{let mut s=p.borrow_mut();s.archive.store(&failed_receipt(1,Refund::InFlight));});
        before_upgrade();
    }
    #[test]
    fn legacy_fee_versions_in_upgrade_metadata_do_not_discard_receipts_or_refunds() {
        configure_test_tariff(10,3);
        let mut completed=failed_receipt(1,Refund::None);
        completed.state=ReceiptState::Completed(InferenceResult{job_id:1,decision:Decision{value:Some("yes".into()),probabilities:vec![0.2,0.7],unknown_probability:0.1,abstained:false,raw_logits:vec![1.,2.,0.],instructions:100,calibration_version:"test".into()},paid_cycles:181,prefix_tokens:5,suffix_tokens:57,workers:vec![]});
        let pending=failed_receipt(2,Refund::Pending);
        PAID.with(|p|{let mut s=p.borrow_mut();s.next_id=2;s.receipts=vec![completed,pending];});
        let expected=metadata();let mut legacy:serde_json::Value=serde_json::from_slice(&expected).unwrap();
        legacy["config"]["version"]=serde_json::json!(7);
        legacy["config"]["enabled"]=serde_json::json!(false);
        for receipt in legacy["receipts"].as_array_mut().unwrap(){receipt["quote"]["version"]=serde_json::json!(6);}
        legacy["receipts"][0]["state"]["Completed"]["quote_version"]=serde_json::json!(6);
        restore_metadata(&serde_json::to_vec(&legacy).unwrap());
        assert_eq!(metadata(),expected);
        PAID.with(|p|{let s=p.borrow();assert_eq!(s.receipts[1].refund,Refund::Pending);assert!(matches!(s.receipts[0].state,ReceiptState::Completed(_)));});
    }
    fn failed_receipt(id:u64, refund:Refund)->Receipt {
        Receipt {caller:Principal::from_slice(&[1,2,3]),request_id:format!("failed-{id}"),input_hash:"a".repeat(64),job_id:id,
            quote:Quote{fee:268_000_000_000,prefix_tokens:38,suffix_tokens:56,estimated_steps:5,max_steps:6},
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
#[ic_cdk::update(name = "preparePaidInferenceUpgradeProbe")]
fn paid_upgrade_probe(active:bool,refund_state:u32){
 let caller=ic_cdk::api::msg_caller();assert!(STORE.with(|s|s.borrow().owner==Some(caller)),"owner only");assert!(refund_state<=3);
 let quote=Quote{fee:1,prefix_tokens:5,suffix_tokens:1,estimated_steps:1,max_steps:6};
 PAID.with(|p|{let mut s=p.borrow_mut();s.active=if active{Some(Job{id:u64::MAX,stage:0,graph_id:None,request:InferRequest{model:"0".repeat(64),version:1,token_ids:COMMON_PREFIX.to_vec(),options:vec!["no".into(),"yes".into()]},quote:quote.clone(),in_flight:false,workers:vec![]})}else{None};
 s.receipts=if refund_state==0{vec![]}else{vec![Receipt{caller,request_id:"diagnostic-upgrade".into(),input_hash:"0".repeat(64),job_id:u64::MAX,quote,time:ic_cdk::api::time(),state:ReceiptState::Failed("diagnostic receipt".into()),refund:match refund_state{1=>Refund::Pending,2=>Refund::InFlight,_=>Refund::Done}}]};});
}
