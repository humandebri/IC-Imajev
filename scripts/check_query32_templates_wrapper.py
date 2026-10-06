#!/usr/bin/env python3
"""Offline CLI integration check against archived real 32-query requests; no new IC measurements."""
import copy,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from full_inference import JournalTransport
from evaluate_prompt_accuracy import base_flags
MODULE='2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05'
D=ROOT/'artifacts/voting-template-prefix-v1/cli-check-v2'
def main():
    D.mkdir(exist_ok=False);rows=[]
    for proposal,record in [('617',0),('620',1),('653',2)]:
        source=ROOT/f'artifacts/voting-template-prefix-v1/proof-v1/{proposal}';reference=json.loads((source/'report.json').read_text());calls=[]
        class Replay(JournalTransport):
            def __init__(self,model,url,canister,pem,directory,pack_hash,wire_codec='',frame_version=1,bridge_binary=None,input_hash=None):
                self.model=model;self.pack_hash=pack_hash;self.wire_codec=wire_codec;self.frame_version=frame_version;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.index=0;self.measurements=[];self.input_hash=input_hash;self.replayed=0
            def close(self):pass
            def command(self,cmd):
                if cmd['op']=='module_hash':return dict(ok=dict(module_hash=MODULE))
                i=self.index;folder=source/'queries';saved=reference['queries'][i]
                assert Path(cmd['input']).read_bytes()==(folder/f'{i:06d}.request.bin').read_bytes(),(proposal,i,'request')
                for key,suffix in [('prefix','prefix.bin'),('expected','expected.bin')]:
                    if key in cmd:assert Path(cmd[key]).read_bytes()==(folder/f'{i:06d}.{suffix}').read_bytes(),(proposal,i,key)
                if cmd['op']=='terminal_step_decision':assert cmd['options']==saved['decision_options']
                for key,suffix in [('output','response.bin'),('hidden','previous.bf16'),('conv','conv.bf16'),('kv','kv.bf16')]:
                    if key in cmd:Path(cmd[key]).write_bytes((folder/f'{i:06d}.{suffix}').read_bytes())
                calls.append(i);return dict(ok=copy.deepcopy(saved['ok']))
        spec=importlib.util.spec_from_file_location('cli',ROOT/'scripts/run_prefix_canister.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.Transport=module.JournalTransport=Replay
        from run_query32_templates import command
        cmd=command('6eydd-o3777-77775-aaama-cai',record,str(D/proposal))[2:]
        sys.argv=['run_prefix_canister.py',*cmd];module.main()
        path=D/proposal/'report.json';r=json.loads(path.read_text());r.update(offline_reconstruction=True,actual_network_query_count=0,instruction_counters_reused_from=str(source.relative_to(ROOT)));path.write_text(json.dumps(r,indent=2)+'\n')
        assert json.loads((D/proposal/'queries/adaptive-plan.json').read_text())['module']==MODULE
        assert r['prefix_bank']==('common' if proposal=='653' else 'voting-fixed-stem38')
        assert calls==list(range(32)) and r['query32_effective'] and r['query_count']==32
        assert np.load(D/proposal/'final-hidden.npy').tobytes()==np.load(source/'final-hidden.npy').tobytes()
        rows.append(dict(proposal=proposal,request_frames_and_auxiliaries_equal=32,network_queries=0,cli_dispatch_verified=True))
    (D/'check.json').write_text(json.dumps(dict(scope=__doc__,cases=rows),indent=2)+'\n')
    print(json.dumps(rows))
if __name__=='__main__':main()
