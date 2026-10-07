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
 graph=graph.replace('front_rows=5120','front_rows=6656').replace('return 5120-layer*256','return 6656-layer*768').replace('[5120,4864,4608,4352]','[6656,5888,5120,4352]')
 (D/'proposal_query32_balanced_graph.py').write_text(graph)
 runner=(D/'runner.py').read_text().replace('proposal_query32_graph','proposal_query32_balanced_graph')
 (D/'balanced-runner.py').write_text(runner)
 plan=json.loads((D/'plan.json').read_text())['records'][15];assert plan['record']==15
 previous=D/'runs/015';archived=D/'failed/015-initial-plan';archived.parent.mkdir(exist_ok=True)
 assert not archived.exists();previous.rename(archived)
 sources=[Path(__file__),D/'proposal_query32_balanced_graph.py',D/'balanced-runner.py']
 o.save(D/'schedule-refinement.json',dict(scope=__doc__,source_hashes={str(p):o.sha(p) for p in sources},initial_attempt=str(archived),fronts=[6656,5888,5120,4352],reason='Original 59 suffix / 38 prefix hit 5B at first MLP-Delta continuation; move more MLP rows into the preceding query.'))
 cache,packets=o.bank(plan['bank']);cmd=o.command(True);cmd[1]=str(D/'balanced-runner.py');cmd[cmd.index('--cache')+1]=str(cache)
 flags=['--hybrid-cache',str(packets),'--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--query32-start','--step-offset','2150000','--record','15','--directory',str(previous)]
 print(json.dumps(dict(stage='balanced-query32-start',fronts=[6656,5888,5120,4352])),flush=True)
 o.call(cmd+flags,previous);print(json.dumps(o.validate(15,previous)),flush=True);o.summarize()

if __name__=='__main__':main()
