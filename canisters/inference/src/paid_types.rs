//! Candid contract shared by the service and its caller example.
use candid::{CandidType,Principal};
use serde::{Deserialize,Serialize};
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct InferRequest {pub model:String,pub version:u32,pub token_ids:Vec<u32>,pub options:Vec<String>}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct Quote {pub version:u64,pub fee:u128,pub prefix_tokens:u32,pub suffix_tokens:u32,pub estimated_steps:u32,pub max_steps:u32}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct Decision {pub value:Option<String>,pub probabilities:Vec<f32>,pub unknown_probability:f32,pub abstained:bool,pub raw_logits:Vec<f32>,pub instructions:u64,pub calibration_version:String}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct WorkerMetric {pub stage:u64,pub instructions:u64,pub heap_pages:u64}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct InferenceResult {pub job_id:u64,pub decision:Decision,pub paid_cycles:u128,pub quote_version:u64,pub prefix_tokens:u32,pub suffix_tokens:u32,pub workers:Vec<WorkerMetric>}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug,PartialEq)]
pub enum Refund {None,Pending,InFlight,Done}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub enum InferError {Invalid(String),NotReady,Busy,Paused,InsufficientCycles{required:u128},QuoteChanged{current:u64},IdConflict,InProgress{job_id:u64},Failed{job_id:u64,reason:String,refund:Refund},ReceiptCapacity}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct StepResult {pub job_id:u64,pub stage:u64,pub done:bool}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct Config {pub enabled:bool,pub version:u64,pub fee_per_token:u128,pub base_fee:u128,pub reserve_cycles:u128}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub enum ReceiptState {Running,Completed(InferenceResult),Failed(String)}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct Receipt {pub caller:Principal,pub request_id:String,pub input_hash:String,pub job_id:u64,pub quote:Quote,pub time:u64,pub state:ReceiptState,pub refund:Refund}
#[derive(CandidType,Deserialize,Serialize,Clone,Debug)]
pub struct DebugResult {pub job_id:u64,pub hidden_hashes:Vec<String>,pub state_hashes:Vec<String>,pub final_hidden:Vec<f32>}
