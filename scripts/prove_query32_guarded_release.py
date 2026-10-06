#!/usr/bin/env python3
"""Replay frozen BOOM queries on a candidate; restore the exact local snapshot."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module

CANISTER = '6eydd-o3777-77775-aaama-cai'
BASELINE = '6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
B = ROOT/'artifacts/guarded-release-v2/full-build'
D = ROOT/'artifacts/guarded-release-v2/query32-proof-v1'
BRIDGE = ROOT/'artifacts/query-packing-v3/build/imajev-client'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temporary = path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    D.mkdir(exist_ok=False)
    manifest = json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    assert manifest['model'] == sha(ROOT/'MODEL_LOCK.json')
    build = json.loads((B/'report.json').read_text())
    candidate = sha(B/'full.wasm')
    assert candidate == build['wasm_sha256'] and build['baseline']==BASELINE
    assert all(p['wasmparser_validation'] for p in build['patches'])
    for key in ['source_hashes','dependency_hashes']:
        assert all(sha(ROOT/p)==h for p,h in build[key].items())
    kernel_check = json.loads((ROOT/'artifacts/delta-register-v3/check/report.json').read_text())
    assert len(kernel_check['cases']) == 7 and all(c['bitwise_equal'] for c in kernel_check['cases'])
    paths = [Path(__file__),ROOT/'client/transport.py',ROOT/'client/prefix_inference.py',BRIDGE,
             ROOT/'scripts/prepare_weight_cache.py',ROOT/'MODEL_LOCK.json',
             ROOT/'checkpoints/full-int8.manifest.json',B/'full.wasm',B/'report.json',
             ROOT/'artifacts/delta-register-v3/build/kernel.wat',
             ROOT/'artifacts/delta-register-v3/check/report.json',ROOT/'scripts/prepare_fixed_prefix_states.py',ROOT/'scripts/prefix_state_args.rs',ROOT/'artifacts/prefix-state-cache-v1/prefix-state-args']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    paths += [ROOT/'client/query32_balanced.py',ROOT/'artifacts/query32-v1/run_balanced.py',ROOT/'artifacts/query32-v1/client-build/imajev-client']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(D/'sources.json',sources)
    with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths: archive.write(path,str(path.relative_to(ROOT)))
    events = []
    def command(operation,*args):
        cmd = ['icp','canister',*operation,CANISTER,*args,'--network','local','--identity','imajev-local']
        start = time.monotonic()
        raw = subprocess.check_output(cmd,text=True,cwd=ROOT)
        events.append(dict(operation=operation,args=list(args),output=raw,seconds=time.monotonic()-start))
        write(D/'operations.json',events)
        return raw
    def bridge(directory):
        return Transport(manifest['model'],'http://localhost:8001/',CANISTER,
                         str(ROOT/'artifacts/imajev-local.pem'),directory,manifest['pack_hash'],bridge_binary=str(BRIDGE))
    t = bridge(D/'before')
    try:
        verify_module(t,BASELINE)
        before_cache = t.command(dict(op='weight_cache_status'))['ok']['cache']
        before_pack = t.command(dict(op='pack_status'))['ok']
        assert before_cache['bytes'] == 4_065_416_192 and len(before_cache['names']) == 721
        assert before_pack['ready'] and before_pack['pack_hash'] == manifest['pack_hash']
    finally: t.close()
    write(D/'before.json',dict(module=BASELINE,cache=before_cache,pack=before_pack))
    snapshot = None
    restored = False
    stopped = False
    results = []
    try:
        command(['stop']); stopped = True
        snapshot = command(['snapshot','create'],'--quiet').strip()
        assert snapshot and all(c in '0123456789abcdefABCDEF' for c in snapshot.removeprefix('0x'))
        write(D/'snapshot.json',dict(canister=CANISTER,snapshot_id=snapshot,baseline=BASELINE,candidate=candidate))
        print(json.dumps(dict(stage='snapshot-saved',snapshot_id=snapshot)),flush=True)
        command(['install'],'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes')
        command(['start']); stopped = False
        print(json.dumps(dict(stage='candidate-installed',module=candidate)),flush=True)
        with (D/'preparation.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_weight_cache.py'),
                '--canister',CANISTER,'--wasm',str(B/'full.wasm'),'--directory',str(D/'preparation'),
                '--include-f32','--require-prepared-rope','--require-prepared-activation','--require-all-output-pairs'],
                cwd=ROOT,check=True,stdout=log,stderr=log)
        preparation = json.loads((D/'preparation/report.json').read_text())
        assert preparation['final'] == before_cache
        print(json.dumps(dict(stage='weights-ready',updates=preparation['update_calls_this_run'])),flush=True)
        with (D/'fixed-prefix-preparation.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_fixed_prefix_states.py'),'--canister',CANISTER,'--wasm',str(B/'full.wasm'),'--directory',str(D/'fixed-prefix-preparation')],cwd=ROOT,check=True,stdout=log,stderr=log)
        fixed_prefix=json.loads((D/'fixed-prefix-preparation/report.json').read_text())
        assert fixed_prefix['update_calls']==24 and fixed_prefix['state_bytes']==50_331_648
        print(json.dumps(dict(stage='fixed-prefix-ready',updates=24,state_bytes=50_331_648)),flush=True)
        t = bridge(D/'replay')
        try:
            verify_module(t,candidate)
            from evaluate_prompt_accuracy import base_flags
            for proposal,record in [('653',2),('620',1)]:
                output=D/proposal
                cmd=base_flags()
                cmd[1]=str(ROOT/'artifacts/query32-v1/run_balanced.py')
                for flag,value in [('--wasm',str(B/'full.wasm')),('--bridge-binary',str(ROOT/'artifacts/query32-v1/client-build/imajev-client')),('--canister',CANISTER),('--reference','artifacts/text-short-v2/inputs.json'),('--cache','artifacts/query-packing-v3/prefix-v2/queries')]:
                    cmd[cmd.index(flag)+1]=value
                cmd += ['--hybrid-cache','artifacts/query-packing-v3/packets-v2','--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start','--roll-start','--packed-start','--record',str(record),'--directory',str(output)]
                with (D/(proposal+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=log)
                result=json.loads((output/'report.json').read_text())
                assert result['query_count']==result['executed_query_count']==32 and result['replayed_queries']==0
                assert result['max_query_instructions']<5_000_000_000
                assert not (output/'queries/failures.jsonl').exists()
                old=ROOT/f'artifacts/boomdao-current-v1/{proposal}-r1'
                baseline=json.loads((old/'report.json').read_text())
                import numpy as np
                assert np.array_equal(np.load(output/'final-hidden.npy'),np.load(old/'final-hidden.npy'))
                for layer in range(31):
                    actual=output/f'queries/layer-{layer:02d}.npy'; reference=old/f'queries/layer-{layer:02d}.npy'
                    if actual.exists() and reference.exists():assert np.array_equal(np.load(actual),np.load(reference)),(proposal,layer)
                a=result['decision_query']['ok']['decision'];b=baseline['decision_query']['ok']['decision']
                assert {k:v for k,v in a.items() if k!='instructions'}=={k:v for k,v in b.items() if k!='instructions'}
                summary=dict(proposal=proposal,queries=32,instructions=result['total_instructions'],max_query_instructions=result['max_query_instructions'],candid_bytes=result['total_candid_bytes'],decision=a,bitwise_equal=True)
                results.append(summary);write(D/'progress.json',results);print(json.dumps(summary),flush=True)
            verify_module(t,candidate)
            assert t.command(dict(op='weight_cache_status'))['ok']['cache'] == before_cache
        finally: t.close()
        assert sources == {str(p.relative_to(ROOT)):sha(p) for p in paths}
        write(D/'report.json',dict(complete=False,baseline=BASELINE,candidate=candidate,source_hashes=sources,
            cases=results,ordinary_queries=sum(r['queries'] for r in results),preparation=preparation,fixed_prefix_preparation=fixed_prefix,
            scope='Actual 32-query graph; full hidden and semantic decision equality.'))
    finally:
        if snapshot is not None:
            if not stopped: command(['stop']); stopped = True
            command(['snapshot','restore'],snapshot)
            command(['start']); stopped = False
            t = bridge(D/'restored')
            try:
                verify_module(t,BASELINE)
                assert t.command(dict(op='weight_cache_status'))['ok']['cache'] == before_cache
                assert t.command(dict(op='pack_status'))['ok'] == before_pack
                restored = True
                write(D/'restored.json',dict(restored=True,module=BASELINE,cache_equal=True,pack_equal=True,snapshot_id=snapshot))
            finally: t.close()
            command(['snapshot','delete'],snapshot)
            print(json.dumps(dict(stage='baseline-restored',cache_equal=True)),flush=True)
        elif stopped:
            command(['start'])
    assert restored and len(results)==2
    report = json.loads((D/'report.json').read_text())
    report.update(complete=True,baseline_snapshot_restored=True,snapshot_deleted=True)
    write(D/'report.json',report)


if __name__ == '__main__':
    main()
