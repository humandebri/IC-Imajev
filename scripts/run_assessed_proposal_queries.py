#!/usr/bin/env python3
"""Repeat an assessed 500-660 task with exact prepared states and measured routing."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import optimize_proposal_queries as o
D=o.D

def generate():
 source=(D/'runner.py').read_text()
 needle=" verifier=Transport(m['model']"
 assert source.count(needle)==1
 affinity=""" if not args.prepare_prefix:
  prefix_source=json.loads((ROOT/args.cache/'cache.json').read_text())['wasm_sha256']
  if prefix_source not in ('6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931','6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4'):raise SystemExit('Unverified prefix module')
  cache_module=prefix_source
"""
 source=source.replace(needle,affinity+needle)
 source=source.replace('import proposal_query32_graph as query32_templates',"import importlib\n    query32_templates=importlib.import_module('proposal_query32_balanced_v3_graph' if args.record==15 else 'proposal_query32_graph')")
 source=source.replace('from proposal_query32_graph import Query32PrefixGraph','Query32PrefixGraph=query32_templates.Query32PrefixGraph')
 destination=D/'final-runner.py'
 if destination.exists():assert destination.read_text()==source
 else:destination.write_text(source)
 files=[Path(__file__),destination,D/'proposal_query32_graph.py',D/'proposal_query32_balanced_v3_graph.py',D/'inputs.json',D/'plan.json']
 identity={str(p):o.sha(p) for p in files}
 metadata=D/'final-runner-identity.json'
 if metadata.exists():assert json.loads(metadata.read_text())==identity
 else:o.save(metadata,identity)
 return destination

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',type=int,choices=range(18));p.add_argument('--directory',type=Path);p.add_argument('--prepare-runner-only',action='store_true');a=p.parse_args()
 runner=generate()
 if a.prepare_runner_only:return
 if a.record is None or a.directory is None:p.error('--record and a fresh --directory are required')
 destination=a.directory.resolve()
 if destination.exists():p.error('output exists; choose a fresh directory')
 plan=json.loads((D/'plan.json').read_text())['records'][a.record];prefix=D/'prefixes'/f"{plan['bank']:02d}";cache=prefix/'queries';packets=D/'packets'/f"{plan['bank']:02d}"
 cmd=o.command(True);cmd[1]=str(runner);cmd[cmd.index('--cache')+1]=str(cache)
 flags=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start']
 if plan['route']=='query32':flags+=['--query32-start']
 flags+=['--step-offset',str(7000000+a.record*10000),'--record',str(a.record),'--directory',str(destination)]
 o.call(cmd+flags,destination);row=o.validate(a.record,destination)
 print(json.dumps(row),flush=True)

if __name__=='__main__':main()
