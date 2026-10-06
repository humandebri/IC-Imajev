#!/usr/bin/env python3
"""Run the verified 32-query route on an already prepared candidate deployment."""
import argparse,json,subprocess
from pathlib import Path
from evaluate_prompt_accuracy import base_flags
ROOT=Path(__file__).resolve().parents[1]
MODULE='2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05'
def command(canister,record,directory):
 cmd=base_flags()
 for flag,value in [('--canister',canister),('--wasm','artifacts/voting-template-prefix-v1/full-build/full.wasm'),('--reference','artifacts/text-short-v2/inputs.json'),('--cache','artifacts/query-packing-v3/prefix-v2/queries'),('--bridge-binary','artifacts/query32-v1/client-build/imajev-client')]:cmd[cmd.index(flag)+1]=value
 return cmd+['--hybrid-cache','artifacts/query-packing-v3/packets-v2','--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--query32-start','--record',str(record),'--directory',directory]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--canister',required=True);ap.add_argument('--record',type=int,choices=[0,1,2],default=0);ap.add_argument('--directory',required=True);a=ap.parse_args()
 import hashlib
 assert hashlib.sha256((ROOT/'artifacts/voting-template-prefix-v1/full-build/full.wasm').read_bytes()).hexdigest()==MODULE
 subprocess.run(command(a.canister,a.record,a.directory),cwd=ROOT,check=True)
 r=json.loads((ROOT/a.directory/'report.json').read_text());assert r['query32_effective'] and r['query_count']==r['executed_query_count']==32 and r['replayed_queries']==0
if __name__=='__main__':main()
