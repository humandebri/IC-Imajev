"""Binary score interpretation and execution of an explicitly frozen runner."""
import json
import math
from pathlib import Path
import subprocess
import sys
from .token_sweep import ROOT, save, sha

IMAJEV = ROOT

def binary_decision(logits, threshold):
    if len(logits)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in logits):
        raise ValueError('two finite raw scores required')
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not .5<=threshold<=1:
        raise ValueError('threshold must be in [0.5,1]')
    first,second = logits
    score = 1 / (1 + math.exp(-abs(first-second)))
    if first == second or score < threshold:
        return 'hold',score
    return ('approve' if first>second else 'reject'),score

def run(out, shard, shards):
    """Compatibility execution for archived local runs; current runs use the fresh driver."""
    if type(shard) is not int or type(shards) is not int or not 0<=shard<shards:
        raise ValueError('invalid shard')
    identity = json.loads((out/'identity.json').read_text())
    checks = {out/'inputs.json':identity['inputs_sha256'], out/'prepared.json':identity['prepared_sha256'],
              out/'runner.py':identity['runner_sha256'], IMAJEV/'target/release/imajev-client':identity['bridge_sha256']}
    source_manifest = out/'source-hashes.json'
    if source_manifest.exists():
        checks.update({Path(path):digest for path,digest in json.loads(source_manifest.read_text()).items()})
    for path,digest in checks.items():
        if sha(path)!=digest:
            raise ValueError('frozen runner/input identity mismatch')
    sys.path.insert(0,str(IMAJEV/'scripts'))
    from evaluate_prompt_accuracy import base_flags
    template = base_flags()
    for flag in ('--cache','--bridge-binary'):
        index = template.index(flag)
        del template[index:index+2]
    template[1] = str(out/'runner.py')
    template[template.index('--reference')+1] = str(out/'inputs.json')
    template.insert(1,'-B')
    for entry in json.loads((out/'prepared.json').read_text())['entries']:
        index = entry['record_index']
        destination = out/'runs'/f'{index:03d}'
        if index%shards!=shard or (destination/'report.json').exists():
            continue
        destination.mkdir(parents=True,exist_ok=True)
        command = template + ['--record',str(index),'--directory',str(destination),'--terminal-readout',
                              '--fuse-terminal-attention','--fuse-terminal-decision']
        save(destination/'command.json',command)
        with (destination/'run.log').open('a') as log:
            process = subprocess.run(command,cwd=IMAJEV,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            save(destination/'error.json',dict(returncode=process.returncode,tail=(destination/'run.log').read_text()[-2500:]))
