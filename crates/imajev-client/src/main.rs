use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use ic_agent::{identity::Secp256k1Identity, Agent};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, BufRead, Write},
    path::Path,
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
        .query(&canister, "pack_status")
        .with_arg(Encode!()?)
        .call()
        .await?;
    let mut status = Decode!(&reply, PackStatus)?;
    if status.model.is_empty() {
        let reply = agent
            .update(&canister, "prepare")
            .with_arg(Encode!(&manifest)?)
            .call_and_wait()
            .await?;
        Decode!(&reply,Result<(),String>)?.map_err(io::Error::other)?;
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
                .update(&canister, "upload_chunk")
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
            .update(&canister, "hash_pack")
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
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let url = &args[1];
    if !url.starts_with("http://localhost:") && !url.starts_with("http://127.0.0.1:") {
        return Err("local only".into());
    }
    let agent = Agent::builder()
        .with_url(url)
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
 "pack_status"=>{
   let b=agent.query(&canister,"pack_status").with_arg(Encode!()?).call().await?;
   let s=Decode!(&b,PackStatus)?;Ok(json!({"model":s.model,"pack_hash":s.pack_hash,"bytes":s.bytes,"received":s.received,"hashed":s.hashed,"ready":s.ready,"chunks":s.chunks}))
 },
 "upload_parallel"=>parallel_upload(&agent,canister,&cmd).await,
 "warm_weights"=>{
   let name=cmd["name"].as_str().ok_or("weight name")?.to_string();let arg=Encode!(&name)?;
   let b=agent.update(&canister,"warm_weights").with_arg(arg.clone()).call_and_wait().await?;
   let info=Decode!(&b,Result<WeightCacheInfo,String>)?.map_err(io::Error::other)?;
   Ok(json!({"cache":info,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "clear_weight_cache"=>{
   let arg=Encode!()?;let b=agent.update(&canister,"clear_weight_cache").with_arg(arg.clone()).call_and_wait().await?;
   let info=Decode!(&b,WeightCacheInfo)?;Ok(json!({"cache":info,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "weight_cache_status"=>{
   let arg=Encode!()?;let b=agent.query(&canister,"weight_cache_status").with_arg(arg.clone()).call().await?;
   let info=Decode!(&b,WeightCacheInfo)?;Ok(json!({"cache":info,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "upload"=>{
   let manifest=fs::read_to_string(cmd["manifest"].as_str().ok_or("manifest")?)?;
   let b=agent.update(&canister,"prepare").with_arg(Encode!(&manifest)?).call_and_wait().await?;Decode!(&b,Result<(),String>)?.map_err(io::Error::other)?;
   let path=Path::new(cmd["pack"].as_str().ok_or("pack")?);let mut file=fs::File::open(path)?;let mut offset=0;let mut updates=1u64;let mut bytes=0u64;
   loop{use std::io::Read;let mut chunk=vec![0;1_000_000];let n=file.read(&mut chunk)?;if n==0{break}chunk.truncate(n);let sha=Sha256::digest(&chunk).to_vec();let arg=Encode!(&offset,&chunk,&sha)?;bytes+=arg.len()as u64;
   let b=agent.update(&canister,"upload").with_arg(arg).call_and_wait().await?;offset=Decode!(&b,Result<u64,String>)?.map_err(io::Error::other)?;updates+=1;
   if updates%100==0{eprintln!("upload {} MB",offset/1_000_000);}
   }
   let b=agent.update(&canister,"seal").with_arg(Encode!()?).call_and_wait().await?;Decode!(&b,Result<(),String>)?.map_err(io::Error::other)?;
   Ok(json!({"uploaded":offset,"update_calls":updates+1,"candid_request_bytes":bytes}))
 },
 "decision"=>{
 let state=fs::read(cmd["input"].as_str().ok_or("input")?)?;let options:Vec<String>=serde_json::from_value(cmd["options"].clone())?;let arg=Encode!(&state,&options)?;let b=agent.query(&canister,match cmd["method"].as_str().unwrap_or("decision") { "decision"=>"decision", "decision_fast"=>"decision_fast", _=>return Err("decision method".into()) }).with_arg(arg.clone()).call().await?;let d=Decode!(&b,Result<ChoiceResult,String>)?.map_err(io::Error::other)?;Ok(json!({"decision":d,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
 "terminal_step_decision"=>{
   let state=fs::read(cmd["input"].as_str().ok_or("input")?)?;
   let options:Vec<String>=serde_json::from_value(cmd["options"].clone())?;
   let arg=Encode!(&state,&options)?;
   let b=agent.query(&canister,"terminal_step_decision").with_arg(arg.clone()).call().await?;
   let result=Decode!(&b,Result<TerminalDecisionMeasurement,String>)?.map_err(io::Error::other)?;
   let m=result.measurement;fs::write(cmd["output"].as_str().ok_or("output")?,&m.state)?;
   Ok(json!({"instructions":m.instructions,"stable_read_bytes":m.stable_read_bytes,
    "heap_pages":m.heap_pages,"stable_pages":m.stable_pages,"request_bytes":arg.len(),
    "reply_bytes":b.len(),"decision":result.decision}))
 },
 "profile"=>{
   let state=fs::read(cmd["input"].as_str().ok_or("input")?)?;let arg=Encode!(&state)?;
   let b=agent.query(&canister,"profile_step").with_arg(arg.clone()).call().await?;
   let p=Decode!(&b,Result<ProfileMeasurement,String>)?.map_err(io::Error::other)?;
   fs::write(cmd["output"].as_str().ok_or("output")?,&p.measurement.state)?;
   Ok(json!({"instructions":p.measurement.instructions,"spans":p.spans,"request_bytes":arg.len(),"reply_bytes":b.len(),"stable_read_bytes":p.measurement.stable_read_bytes}))
 },
 "step"=>{
   let state=fs::read(cmd["input"].as_str().ok_or("input")?)?;let arg=Encode!(&state)?;let b=agent.query(&canister,"step").with_arg(arg.clone()).call().await?;
   let m=Decode!(&b,Result<Measurement,String>)?.map_err(io::Error::other)?;fs::write(cmd["output"].as_str().ok_or("output")?,&m.state)?;
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
