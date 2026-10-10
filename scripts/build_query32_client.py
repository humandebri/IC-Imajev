#!/usr/bin/env python3
"""Build the frozen bridge with the Attention -> MLP-front query operation."""
import hashlib,json,os,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/query32-v1/client-build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=False);old=ROOT/'artifacts/query-packing-v3/build';source=(old/'client.rs').read_text()
 marker='#[derive(CandidType, Deserialize)]\nstruct ProfileMeasurement'
 definition='#[derive(CandidType, Deserialize)]\nstruct AttentionMlpFrontMeasurement {state:Vec<u8>,previous_hidden:Vec<u8>,kv:Vec<u8>,instructions:u64,stable_read_bytes:u64,heap_pages:u64,stable_pages:u64,spans:Vec<(String,u64)>}\n'
 assert source.count(marker)==1;source=source.replace(marker,definition+marker)
 marker=' "mlp_delta_front"=>{';assert source.count(marker)==1
 branch=''' "attention_mlp_front"=>{
  let(state,_)=read_inference_state(cmd["input"].as_str().ok_or("input")?)?;
  let(_,bound)=read_inference_state(cmd["expected"].as_str().ok_or("expected reply identity")?)?;
  let front=u32::try_from(cmd["front"].as_u64().ok_or("front")?)?;
  let arg=Encode!(&state,&front)?;let b=agent.query(&canister,"runAttentionInferenceStep").with_arg(arg.clone()).call().await?;
  let m=Decode!(&b,Result<AttentionMlpFrontMeasurement,String>)?.map_err(io::Error::other)?;
  store_inference_reply(cmd["output"].as_str().ok_or("output")?,m.state,bound)?;
  fs::write(cmd["hidden"].as_str().ok_or("hidden output")?,m.previous_hidden)?;fs::write(cmd["kv"].as_str().ok_or("kv output")?,m.kv)?;
  Ok(json!({"instructions":m.instructions,"stable_read_bytes":m.stable_read_bytes,"heap_pages":m.heap_pages,"stable_pages":m.stable_pages,"spans":m.spans,"request_bytes":arg.len(),"reply_bytes":b.len()}))
 },
'''
 source=source.replace(marker,branch+marker);(D/'client.rs').write_text(source)
 cmd=json.loads((old/'client-build.json').read_text())['command'];cmd[cmd.index('--edition=2021')+1]=str(D/'client.rs');cmd[cmd.index('-o')+1]=str(D/'imajev-client')
 sources=[D/'client.rs',Path(__file__),old/'client.rs',old/'client-build.json'];refs={str(p.relative_to(ROOT)):sha(p) for p in sources};deps={}
 for i,arg in enumerate(cmd):
  if i and cmd[i-1]=='--extern':path=Path(arg.split('=',1)[1]);deps[str(path.relative_to(ROOT))]=sha(path)
 with (D/'compiler.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CARGO_PKG_VERSION='0.1.0'),stdout=log,stderr=log,check=True)
 assert refs=={p:sha(ROOT/p) for p in refs};assert deps=={p:sha(ROOT/p) for p in deps}
 (D/'report.json').write_text(json.dumps(dict(command=cmd,source_hashes=refs,dependency_hashes=deps,bridge_sha256=sha(D/'imajev-client')),indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sources:z.write(p,str(p.relative_to(ROOT)))
 print(sha(D/'imajev-client'))
if __name__=='__main__':main()
