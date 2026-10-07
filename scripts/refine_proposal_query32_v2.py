#!/usr/bin/env python3
"""Rebalance a measured failing schedule without changing arithmetic or inputs."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import optimize_proposal_queries as o
D=o.D

def main():
 graph=(D/'proposal_query32_graph.py').read_text()
 assert 'front_rows=5120' in graph and '[5120,4864,4608,4352]' in graph
 graph=graph.replace('front_rows=5120','front_rows=6400').replace('return 5120-layer*256','return [6400,5888,5120][layer]').replace('[5120,4864,4608,4352]','[6400,5888,5120,4352]')
 (D/'proposal_query32_balanced_v2_graph.py').write_text(graph)
 runner=(D/'runner.py').read_text().replace('proposal_query32_graph','proposal_query32_balanced_v2_graph')
 (D/'balanced-v2-runner.py').write_text(runner)
 plan=json.loads((D/'plan.json').read_text())['records'][15];assert plan['record']==15
 previous=D/'runs/015';archived=D/'failed/015-balanced-v1';archived.parent.mkdir(exist_ok=True)
 assert not archived.exists();previous.rename(archived)
 sources=[Path(__file__),D/'proposal_query32_balanced_v2_graph.py',D/'balanced-v2-runner.py']
 o.save(D/'schedule-refinement-v2.json',dict(scope=__doc__,source_hashes={str(p):o.sha(p) for p in sources},initial_attempt=str(archived),fronts=[6400,5888,5120,4352],reason='First rebalanced plan passed Delta queries but exceeded 5B at Attention7; reduce next front by 256 while keeping Delta under budget.'))
 cache,packets=o.bank(plan['bank']);cmd=o.command(True);cmd[1]=str(D/'balanced-v2-runner.py');cmd[cmd.index('--cache')+1]=str(cache)
 flags=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--query32-start','--step-offset','3150000','--record','15','--directory',str(previous)]
 print(json.dumps(dict(stage='balanced-query32-start',fronts=[6400,5888,5120,4352])),flush=True)
 o.call(cmd+flags,previous);print(json.dumps(o.validate(15,previous)),flush=True);o.summarize()

if __name__=='__main__':main()
