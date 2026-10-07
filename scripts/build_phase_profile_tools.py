#!/usr/bin/env python3
"""Decode saved phase-profile Candid String replies into typed phase rows."""
from pathlib import Path
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/phase-profile-tools-v1';d.mkdir(exist_ok=False);old=json.loads((ROOT/'artifacts/paid-update-v1/tools/report.json').read_text());assert all(sha(ROOT/p)==h for p,h in old['source_hashes'].items());deps={str(Path(arg.split('=',1)[1]).relative_to(ROOT)):sha(Path(arg.split('=',1)[1]))for i,arg in enumerate(old['args_command'])if i>0 and old['args_command'][i-1]=='--extern'}
 source='''use candid::Decode;
fn main(){let path=std::env::args().nth(1).unwrap();let text=std::fs::read_to_string(path).unwrap();let clean:String=text.trim().trim_start_matches("0x").chars().filter(|c|!c.is_whitespace()).collect();assert_eq!(clean.len()%2,0);let bytes:Vec<u8>=(0..clean.len()).step_by(2).map(|i|u8::from_str_radix(&clean[i..i+2],16).unwrap()).collect();let value=Decode!(&bytes,String).unwrap();let rows:Vec<(String,u64,u64)>=serde_json::from_str(&value).unwrap();println!("{}",serde_json::to_string(&rows).unwrap());}
'''
 p=d/'decode.rs';p.write_text(source);cmd=old['args_command'][:];cmd[cmd.index('--edition=2021')+1]=str(p);cmd[cmd.index('-o')+1]=str(d/'decode')
 with(d/'compiler.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log,check=True)
 (d/'report.json').write_text(json.dumps(dict(command=cmd,source_hashes={str(p.relative_to(ROOT)):sha(p),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},dependency_hashes=deps,binary_sha256=sha(d/'decode')),indent=2)+'\n')
 print('typed native phase decoder built')
if __name__=='__main__':main()
