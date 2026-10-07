//! Owner-controlled example relay: attaches cycles and measures returned funds.
use candid::{CandidType,Deserialize,Principal};
use std::cell::RefCell;
use ic_cdk::call::Call;
thread_local!{static OWNER:RefCell<Option<Principal>>=RefCell::new(None);}
#[ic_cdk::init]fn init(owner:Principal){assert_ne!(owner,Principal::anonymous());OWNER.with(|s|*s.borrow_mut()=Some(owner));}
fn guard()->Result<(),String>{if OWNER.with(|s|*s.borrow())==Some(ic_cdk::api::msg_caller()){Ok(())}else{Err("owner only".into())}}
#[derive(CandidType,Deserialize)]struct Forward {target:Principal,method:String,args:Vec<u8>,cycles:u128}
#[derive(CandidType,Deserialize)]struct ForwardResult {response:Result<Vec<u8>,String>,refunded:u128,balance_before:u128,balance_after:u128,instructions:u64}
#[ic_cdk::update(guard="guard")]
async fn forward(request:Forward)->ForwardResult {
 let before=ic_cdk::api::canister_cycle_balance();let started=ic_cdk::api::call_context_instruction_counter();
 let result=Call::unbounded_wait(request.target,&request.method).take_raw_args(request.args).with_cycles(request.cycles).await;
 let refunded=ic_cdk::api::msg_cycles_refunded();let response=result.map(|v|v.into_bytes()).map_err(|e|e.to_string());
 ForwardResult{response,refunded,balance_before:before,balance_after:ic_cdk::api::canister_cycle_balance(),instructions:ic_cdk::api::call_context_instruction_counter()-started}
}
#[ic_cdk::query(guard="guard")]fn balance()->u128{ic_cdk::api::canister_cycle_balance()}
#[ic_cdk::pre_upgrade]fn pre_upgrade(){ic_cdk::storage::stable_save((OWNER.with(|s|*s.borrow()),)).expect("owner metadata");}
#[ic_cdk::post_upgrade]fn post_upgrade(){let (o,):(Option<Principal>,)=ic_cdk::storage::stable_restore().expect("owner metadata");OWNER.with(|s|*s.borrow_mut()=o);}
ic_cdk::export_candid!();
