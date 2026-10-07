#!/usr/bin/env python3
"""Finish on an unchanged prepared comparison module; do not touch other work."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import optimize_proposal_queries as o
D=o.D

def main():
 runner=(D/'runner.py').read_text()
 needle="cache_module='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'"
 assert runner.count(needle)==1
 runner=runner.replace(needle,f"cache_module='{o.CANDIDATE}'")
 (D/'same-module-runner.py').write_text(runner)
 o.save(D/'comparison-affinity.json',dict(module=o.CANDIDATE,canister='2hlkr-gl777-77775-aaaxa-cai',scope=__doc__,sources={str(p):o.sha(p) for p in [Path(__file__),D/'same-module-runner.py']}))
 plan=json.loads((D/'plan.json').read_text())
 for i in (16,17):
  p=plan['records'][i];dest=D/'runs'/f'{i:03d}'
  if (dest/'verified.json').exists():continue
  prefix=D/'prefixes'/f"{p['bank']:02d}";packets=D/'packets'/f"{p['bank']:02d}";cache=prefix/'queries'
  if prefix.exists() and not (prefix/'report.json').exists():
   archived=D/'failed'/f"prefix-{p['bank']:02d}-module-mismatch";prefix.rename(archived)
  if not (prefix/'report.json').exists():
   cmd=o.command(True);cmd[1]=str(D/'same-module-runner.py');cmd[cmd.index('--cache')+1]=str(cache)
   print(json.dumps(dict(stage='comparison-prefix-start',record=i,tokens=p['prefix'])),flush=True)
   o.call(cmd+['--record',str(i),'--prefix-tokens',str(p['prefix']),'--prepare-prefix','--directory',str(prefix)],prefix)
  if not (packets/'cache.json').exists():
   cmd=[sys.executable,'-B',str(ROOT/'scripts/prepare_prefix_reuse.py'),'--prefix-directory',str(prefix),'--directory',str(packets),'--canister','2vn5i-k3777-77775-aaaua-cai','--codec-module','e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e','--run-report',str(prefix/'codec-report.json')]
   with (prefix/'codec.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log,check=True)
  cmd=o.command(True);cmd[1]=str(D/'same-module-runner.py');cmd[cmd.index('--cache')+1]=str(cache)
  flags=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start']
  if p['route']=='query32':flags+=['--query32-start']
  flags+=['--step-offset',str(5160000+i*10000),'--record',str(i),'--directory',str(dest)]
  print(json.dumps(dict(stage='comparison-inference-start',record=i,route=p['route'])),flush=True)
  o.call(cmd+flags,dest);print(json.dumps(o.validate(i,dest)),flush=True);o.summarize()

if __name__=='__main__':main()
