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
B = ROOT/'artifacts/update-templates-v1/build'
D = ROOT/'artifacts/update-templates-v1/proof-v2'
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
    paths = [Path(__file__),ROOT/'scripts/measure_update_templates.py',ROOT/'client/transport.py',ROOT/'client/prefix_inference.py',BRIDGE,
             ROOT/'scripts/prepare_weight_cache.py',ROOT/'MODEL_LOCK.json',
             ROOT/'checkpoints/full-int8.manifest.json',B/'full.wasm',B/'report.json',
             ROOT/'artifacts/delta-register-v3/build/kernel.wat',
             ROOT/'artifacts/delta-register-v3/check/report.json',ROOT/'scripts/prepare_fixed_prefix_states.py',ROOT/'scripts/prefix_state_args.rs',ROOT/'artifacts/prefix-state-cache-v1/prefix-state-args']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    paths += [ROOT/'artifacts/voting-template-prefix-v1/template.json',ROOT/'artifacts/voting-template-prefix-v1/prefix/report.json',ROOT/'artifacts/voting-template-prefix-v1/prefix/queries/cache.json',ROOT/'artifacts/voting-template-prefix-v1/packets/cache.json',ROOT/'client/query32_templates.py',ROOT/'artifacts/voting-template-prefix-v1/run_query32.py',ROOT/'artifacts/query32-v1/client-build/imajev-client']
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
        for bank in ['voting','common']:
            if bank=='common':
                command(['stop']); stopped=True
                command(['install'],'--mode','upgrade','--wasm',str(B/'full.wasm'),'--yes')
                command(['start']); stopped=False
            bankdir=D/bank;bankdir.mkdir()
            with (bankdir/'preparation.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/prepare_weight_cache.py'),
                    '--canister',CANISTER,'--wasm',str(B/'full.wasm'),'--directory',str(bankdir/'preparation'),
                    '--include-f32','--require-prepared-rope','--require-prepared-activation','--require-all-output-pairs'],
                    cwd=ROOT,check=True,stdout=log,stderr=log)
            preparation=json.loads((bankdir/'preparation/report.json').read_text())
            assert preparation['final']==before_cache
            print(json.dumps(dict(stage='weights-ready',bank=bank,updates=preparation['update_calls_this_run'])),flush=True)
            with (bankdir/'fixed-prefix-preparation.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/prepare_fixed_prefix_states.py'),'--canister',CANISTER,'--wasm',str(B/'full.wasm'),'--directory',str(bankdir/'fixed-prefix-preparation')],cwd=ROOT,check=True,stdout=log,stderr=log)
            fixed_prefix=json.loads((bankdir/'fixed-prefix-preparation/report.json').read_text())
            assert fixed_prefix['update_calls']==24
            with (bankdir/'measurement.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/measure_update_templates.py'),'--wasm',str(B/'full.wasm'),'--directory',str(bankdir/'measurement'),'--bank',bank,'--prepare-prefix','--repeats','3'],cwd=ROOT,check=True,stdout=log,stderr=log)
            result=json.loads((bankdir/'measurement/report.json').read_text());assert result['complete']
            results.append(dict(bank=bank,preparation=preparation,fixed_prefix_preparation=fixed_prefix,measurement=result))
            for row in result['cases']:
                print(json.dumps(dict(case=row['case'],repeat=row['repeat'],updates=row['update_calls'],instructions=row['total_handler_instructions'],seconds=row['call_wall_seconds'])),flush=True)
        assert sources == {str(p.relative_to(ROOT)):sha(p) for p in paths}
        write(D/'report.json',dict(complete=False,baseline=BASELINE,candidate=candidate,source_hashes=sources,
            banks=results,inference_runs=sum(len(r['measurement']['cases']) for r in results),
            inference_update_calls=sum(c['update_calls'] for r in results for c in r['measurement']['cases']),
            scope='Actual server-held updates with optimized kernels and voting38/common27 prefixes; unchanged 34B scheduler.'))
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
