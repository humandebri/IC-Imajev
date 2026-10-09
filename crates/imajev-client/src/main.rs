use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use ic_agent::{identity::Secp256k1Identity, Agent};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, BufRead, Write},
    time::Instant,
};
#[derive(CandidType, Deserialize)]
struct Measurement {
    state: Vec<u8>,
    instructions: u64,
    stable_read_bytes: u64,
    heap_pages: u64,
    stable_pages: u64,
}
#[derive(CandidType, Deserialize)]
struct MlpDeltaMeasurement { state:Vec<u8>, previous_hidden:Vec<u8>, conv:Vec<u8>, instructions:u64, stable_read_bytes:u64, heap_pages:u64, stable_pages:u64, spans:Vec<(String,u64)> }
#[derive(CandidType, Deserialize)]
struct ProfileMeasurement {
    measurement: Measurement,
    spans: Vec<(String, u64, u64)>,
}
#[derive(CandidType, Deserialize)]
struct TerminalDecisionMeasurement { measurement: Measurement, decision: ChoiceResult }
#[derive(CandidType, Deserialize, serde::Serialize)]
struct ChoiceResult {
    value: Option<String>,
    probabilities: Vec<f32>,
    unknown_probability: f32,
    abstained: bool,
    raw_logits: Vec<f32>,
    instructions: u64,
    calibration_version: String,
}
#[derive(CandidType, Deserialize, serde::Serialize)]
struct WeightCacheInfo {
    bytes: u64,
    names: Vec<String>,
    preparation_instructions: u64,
    rope_bytes: Option<u64>,
    activation_bytes: Option<u64>,
    paired_weight_bytes: Option<u64>,
}
#[derive(CandidType, Deserialize)]
struct PackStatus {
    model: String,
    pack_hash: String,
    bytes: u64,
    received: u64,
    hashed: u64,
    ready: bool,
    chunks: Vec<u64>,
}
async fn parallel_upload(
    agent: &Agent,
    canister: Principal,
    cmd: &Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};
    let manifest = fs::read_to_string(cmd["manifest"].as_str().ok_or("manifest")?)?;
    let m: imajev_runtime::Manifest = serde_json::from_str(&manifest)?;
    let reply = agent
        .query(&canister, "getModelStatus")
        .with_arg(Encode!()?)
        .call()
        .await?;
    let mut status = Decode!(&reply, PackStatus)?;
    if status.model.is_empty() || cmd["reset"].as_bool() == Some(true) {
        let reply = agent
            .update(&canister, "prepareModelUpload")
            .with_arg(Encode!(&manifest)?)
            .call_and_wait()
            .await?;
        Decode!(&reply,Result<(),String>)?.map_err(io::Error::other)?;
        status.chunks.clear();
        status.received = 0;
        status.hashed = 0;
        status.ready = false;
        status.model = m.model.clone();
        status.pack_hash = m.pack_hash.clone();
        status.bytes = m.bytes;
    }
    if status.model != m.model || status.pack_hash != m.pack_hash || status.bytes != m.bytes {
        return Err("existing pack mismatch".into());
    }
    if status.ready {
        return Ok(json!({"already_ready":true,"uploaded":status.bytes}));
    }
    let completed: std::collections::BTreeSet<_> = status.chunks.into_iter().collect();
    let mut file = fs::File::open(cmd["pack"].as_str().ok_or("pack")?)?;
    if file.metadata()?.len() != m.bytes {
        return Err("pack length".into());
    }
    let concurrency = cmd["concurrency"].as_u64().unwrap_or(8).clamp(1, 16) as usize;
    let mut jobs = tokio::task::JoinSet::new();
    let mut count = 0u64;
    let mut sent = 0u64;
    let mut updates = 0u64;
    for offset in (0..m.bytes).step_by(1_800_000) {
        if completed.contains(&offset) {
            continue;
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; (m.bytes - offset).min(1_800_000) as usize];
        file.read_exact(&mut bytes)?;
        let sha = Sha256::digest(&bytes).to_vec();
        let arg = Encode!(&offset, &bytes, &sha)?;
        sent += arg.len() as u64;
        let agent = agent.clone();
        jobs.spawn(async move {
            let reply = agent
                .update(&canister, "uploadModelChunk")
                .with_arg(arg)
                .call_and_wait()
                .await
                .map_err(|e| e.to_string())?;
            Decode!(&reply,Result<u64,String>).map_err(|e| e.to_string())?
        });
        if jobs.len() >= concurrency {
            jobs.join_next().await.unwrap()?.map_err(io::Error::other)?;
            count += 1;
            updates += 1;
        }
        if count > 0 && count % 64 == 0 {
            eprintln!("uploaded chunks {count}, sent {} MB", sent / 1_000_000);
        }
    }
    while let Some(job) = jobs.join_next().await {
        job?.map_err(io::Error::other)?;
        updates += 1;
    }
    eprintln!("chunks uploaded; hashing pack");
    loop {
        let reply = agent
            .update(&canister, "verifyModelUpload")
            .with_arg(Encode!(&8_000_000u64)?)
            .call_and_wait()
            .await?;
        let (hashed, ready) =
            Decode!(&reply,Result<(u64,bool),String>)?.map_err(io::Error::other)?;
        updates += 1;
        if updates % 64 == 0 {
            eprintln!("hashed {} / {} MB", hashed / 1_000_000, m.bytes / 1_000_000);
        }
        if ready {
            break;
        }
    }
    Ok(
        json!({"uploaded":m.bytes,"update_calls_this_run":updates,"candid_chunk_request_bytes_this_run":sent,"concurrency":concurrency}),
    )
}
// Version3 is checked before signing the outgoing message. Seal only after
// Agent::query().call() has verified the incoming node signature.
fn read_inference_state(path:&str)->Result<(Vec<u8>,Option<imajev_runtime::HostBoundRequest>),Box<dyn std::error::Error>> {
    let b=fs::read(path)?;
    let bound=if imajev_runtime::is_host_bound_frame(&b){Some(imajev_runtime::HostBoundRequest::verify_stored(&b).map_err(io::Error::other)?)}else{None};Ok((b,bound))
}
fn store_inference_reply(path:&str,state:Vec<u8>,bound:Option<imajev_runtime::HostBoundRequest>)->Result<(),Box<dyn std::error::Error>> {
    let state=if let Some(bound)=bound{bound.seal_verified_reply(state).map_err(io::Error::other)?}else{state};fs::write(path,state)?;Ok(())
}
// Replicated ingress calls to the existing query exports, for local A/B measurement.
async fn inference_call(agent: &Agent, canister: Principal, method: &str, arg: Vec<u8>, cmd: &Value) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match cmd["execution"].as_str().unwrap_or("query") {
        "query" => Ok(agent.query(&canister, method).with_arg(arg).call().await?),
        "update" => Ok(agent.update(&canister, method).with_arg(arg).call_and_wait().await?),
        _ => Err("execution must be query or update".into()),
    }
}
#[derive(CandidType)]
struct StatusArgs { canister_id: Principal }
#[derive(CandidType, Deserialize)]
struct BalanceStatus { cycles: candid::Nat, memory_size: candid::Nat, module_hash: Option<Vec<u8>> }
#[derive(CandidType, Deserialize, serde::Serialize)]
struct UpdateProgress {
    id:u64, stage:u64, done:bool, instructions:u64, stable_read_bytes:u64,
    operations:Vec<(String,u64)>, hidden_hashes:Vec<String>, state_hashes:Vec<String>,
    final_hidden:Vec<f32>, decision:Option<ChoiceResult>, heap_pages:u64,
}
fn require_diagnostics(cmd: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if cmd["diagnostics"].as_bool() != Some(true) {
        return Err("diagnostic operation requires diagnostics:true and a paid-update-diagnostics canister".into());
    }
    Ok(())
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let url = &args[1];
    if !url.starts_with("http://localhost:") && !url.starts_with("http://127.0.0.1:") {
        return Err("local only".into());
    }
    let agent = Agent::builder()
        .with_url(url)
        .with_verify_query_signatures(true)
        .with_identity(Secp256k1Identity::from_pem_file(&args[3])?)
        .build()?;
    agent.fetch_root_key().await?;
    let canister = Principal::from_text(&args[2])?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let cmd: Value = serde_json::from_str(&line)?;
        let start = Instant::now();
        let result:Result<Value,Box<dyn std::error::Error>>=async {
 match cmd["op"].as_str().ok_or("op")?{
 "module_hash"=>{ let hash=agent.read_state_canister_module_hash(canister).await?; Ok(json!({"module_hash":hash.iter().map(|b|format!("{b:02x}")).collect::<String>()})) },
 "update_prefix"=>{
   let layer=cmd["layer"].as_u64().ok_or("layer")? as u32;
   let raw=fs::read(cmd["values"].as_str().ok_or("values")?)?;
   if raw.len()%4!=0{return Err("float file length".into());}
   let values:Vec<f32>=raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
   let packet=if let Some(path)=cmd["packet"].as_str(){fs::read(path)?}else{vec![]};
   let arg=Encode!(&layer,&values,&packet)?;
   let b=agent.update(&canister,"installInferencePrefix").with_arg(arg.clone()).call_and_wait().await?;
   Decode!(&b,Result<(),String>)?.map_err(io::Error::other)?;
   Ok(json!({"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "update_infer_start"|"update_infer_continue"=>{
   require_diagnostics(&cmd)?;
   let op=cmd["op"].as_str().unwrap();
   let arg=if op=="update_infer_start" {let ids:Vec<u32>=serde_json::from_value(cmd["ids"].clone())?;let options:Vec<String>=serde_json::from_value(cmd["options"].clone())?;Encode!(&ids,&options)?}
     else {let id=cmd["id"].as_u64().ok_or("id")?;let stage=cmd["stage"].as_u64().ok_or("stage")?;Encode!(&id,&stage)?};
   let b=agent.update(&canister,if op=="update_infer_start" {"startOwnerInference"} else {"continueOwnerInference"}).with_arg(arg.clone()).call_and_wait().await?;
   let result=Decode!(&b,Result<UpdateProgress,String>)?.map_err(io::Error::other)?;
   Ok(json!({"progress":result,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "balance_status"=>{
   let arg=Encode!(&StatusArgs { canister_id: canister })?;
   let b=agent.update(&Principal::management_canister(),"canister_status").with_effective_canister_id(canister).with_arg(arg).call_and_wait().await?;
   let s=Decode!(&b,BalanceStatus)?;
   Ok(json!({"cycles":s.cycles.0.to_string(),"memory_size":s.memory_size.0.to_string(),"module_hash":s.module_hash.map(|h|h.iter().map(|b|format!("{b:02x}")).collect::<String>())}))
 },
 "pack_status"=>{
   let b=agent.query(&canister,"getModelStatus").with_arg(Encode!()?).call().await?;
   let s=Decode!(&b,PackStatus)?;Ok(json!({"model":s.model,"pack_hash":s.pack_hash,"bytes":s.bytes,"received":s.received,"hashed":s.hashed,"ready":s.ready,"chunks":s.chunks}))
 },
 "upload_parallel"=>parallel_upload(&agent,canister,&cmd).await,
 "warm_weights"=>{
   let name=cmd["name"].as_str().ok_or("weight name")?.to_string();let arg=Encode!(&name)?;
   let b=agent.update(&canister,"prepareWeightCache").with_arg(arg.clone()).call_and_wait().await?;
   let info=Decode!(&b,Result<WeightCacheInfo,String>)?.map_err(io::Error::other)?;
   Ok(json!({"cache":info,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "clear_weight_cache"=>{
   let arg=Encode!()?;let b=agent.update(&canister,"clearWeightCache").with_arg(arg.clone()).call_and_wait().await?;
   let info=Decode!(&b,WeightCacheInfo)?;Ok(json!({"cache":info,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "weight_cache_status"=>{
   let arg=Encode!()?;let b=agent.query(&canister,"getWeightCacheStatus").with_arg(arg.clone()).call().await?;
   let info=Decode!(&b,WeightCacheInfo)?;Ok(json!({"cache":info,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "upload"=>{
   let mut sequential=cmd.clone();sequential["concurrency"]=json!(1);
   parallel_upload(&agent,canister,&sequential).await
 },
 "terminal_step_decision"=>{
   let (state,bound)=read_inference_state(cmd["input"].as_str().ok_or("input")?)?;
   let options:Vec<String>=serde_json::from_value(cmd["options"].clone())?;
   let arg=Encode!(&state,&options)?;
   let b=inference_call(&agent,canister,"runFinalInferenceStep",arg.clone(),&cmd).await?;
   let result=Decode!(&b,Result<TerminalDecisionMeasurement,String>)?.map_err(io::Error::other)?;
   let m=result.measurement;store_inference_reply(cmd["output"].as_str().ok_or("output")?,m.state,bound)?;
   Ok(json!({"instructions":m.instructions,"stable_read_bytes":m.stable_read_bytes,
    "heap_pages":m.heap_pages,"stable_pages":m.stable_pages,"request_bytes":arg.len(),
    "reply_bytes":b.len(),"decision":result.decision}))
 },
 "profile"=>{
   require_diagnostics(&cmd)?;
   let (state,bound)=read_inference_state(cmd["input"].as_str().ok_or("input")?)?;let arg=Encode!(&state)?;
   let b=agent.query(&canister,"profileInferenceStep").with_arg(arg.clone()).call().await?;
   let p=Decode!(&b,Result<ProfileMeasurement,String>)?.map_err(io::Error::other)?;
   store_inference_reply(cmd["output"].as_str().ok_or("output")?,p.measurement.state,bound)?;
   Ok(json!({"instructions":p.measurement.instructions,"spans":p.spans,"request_bytes":arg.len(),"reply_bytes":b.len(),"stable_read_bytes":p.measurement.stable_read_bytes}))
 },
 "mlp_delta_front"=>{
  let(state,_)=read_inference_state(cmd["input"].as_str().ok_or("input")?)?;
  let(_,bound)=read_inference_state(cmd["expected"].as_str().ok_or("expected reply identity")?)?;
  let prefix=fs::read(cmd["prefix"].as_str().ok_or("prefix")?)?;
  let p=cmd["prefix_tokens"].as_u64().ok_or("prefix tokens")? as u32;let front=cmd["front"].as_u64().ok_or("front")? as u32;
  let arg=Encode!(&state,&prefix,&p,&front)?;let b=agent.query(&canister,"runDeltaInferenceStep").with_arg(arg.clone()).call().await?;
  let m=Decode!(&b,Result<MlpDeltaMeasurement,String>)?.map_err(io::Error::other)?;
  store_inference_reply(cmd["output"].as_str().ok_or("output")?,m.state,bound)?;
  fs::write(cmd["hidden"].as_str().ok_or("hidden output")?,m.previous_hidden)?;fs::write(cmd["conv"].as_str().ok_or("conv output")?,m.conv)?;
  Ok(json!({"instructions":m.instructions,"stable_read_bytes":m.stable_read_bytes,"heap_pages":m.heap_pages,"stable_pages":m.stable_pages,"spans":m.spans,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "step"=>{
   let (state,bound)=read_inference_state(cmd["input"].as_str().ok_or("input")?)?;let arg=Encode!(&state)?;let b=inference_call(&agent,canister,"runInferenceStep",arg.clone(),&cmd).await?;
   let m=Decode!(&b,Result<Measurement,String>)?.map_err(io::Error::other)?;store_inference_reply(cmd["output"].as_str().ok_or("output")?,m.state,bound)?;
   Ok(json!({"instructions":m.instructions,"stable_read_bytes":m.stable_read_bytes,"heap_pages":m.heap_pages,"stable_pages":m.stable_pages,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 _=>Err("unknown op".into())
 }
 }.await;
        let value = match result {
            Ok(v) => json!({"ok":v,"wall_seconds":start.elapsed().as_secs_f64()}),
            Err(e) => json!({"error":e.to_string(),"wall_seconds":start.elapsed().as_secs_f64()}),
        };
        println!("{value}");
        io::stdout().flush()?;
    }
    Ok(())
}
