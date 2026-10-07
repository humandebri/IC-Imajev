#!/usr/bin/env python3
import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'artifacts/paid-update-v1/tools'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False);build=json.loads((R/'artifacts/update-templates-v1/build/report.json').read_text());cmd=build['command'][:];cmd[cmd.index('--edition=2021')+1]=str(R/'examples/paid-inference-caller/lib.rs');cmd[cmd.index('-o')+1]=str(D/'caller.wasm');cmd[cmd.index('--crate-name')+1]='paid_inference_caller'
 env=dict(os.environ,CARGO_MANIFEST_DIR=str(D),CARGO_PKG_NAME='paid-inference-caller',CARGO_PKG_VERSION='0.1.0',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='paid_inference_caller')
 with (D/'caller-compiler.log').open('w')as log:subprocess.run(cmd,cwd=R,env=env,check=True,stdout=log,stderr=log)
 deps=R/'artifacts/update-inference/client-target/release/deps';native=['rustc','--edition=2021',str(R/'scripts/paid_update_args.rs'),'-C','opt-level=2','-C','panic=abort','-C','lto=thin','-o',str(D/'args'),'-L','dependency='+str(deps)]
 for name in ['candid','serde','serde_json']:
  choices=list(deps.glob('lib'+name+'-*.rlib'));assert len(choices)==1,(name,choices);native+=['--extern',name+'='+str(choices[0])]
 with (D/'args-compiler.log').open('w')as log:subprocess.run(native,cwd=R,check=True,stdout=log,stderr=log)
 paths=[Path(__file__),R/'examples/paid-inference-caller/lib.rs',R/'scripts/paid_update_args.rs',R/'canisters/inference/src/paid_types.rs',D/'caller.wasm',D/'args'];report=dict(source_hashes={str(p.relative_to(R)):sha(p) for p in paths},caller_command=cmd,args_command=native)
 (D/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('caller and Candid tools ready')
if __name__=='__main__':main()
