#!/usr/bin/env python3
"""Run full local query inference and compare exact prefix reuse to saved baseline.

Requires an explicitly installed, uploaded and prepared experimental canister.
This driver does not install/upgrade/upload or change any existing canister.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--canister', required=True)
    ap.add_argument('--codec-canister', required=True)
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--terminal-attention', action='store_true')
    ap.add_argument('--terminal-decision', action='store_true')
    ap.add_argument('--terminal-tail', action='store_true')
    ap.add_argument('--host-checksum',action='store_true')
    ap.add_argument('--prefix-start', action='store_true')
    args = ap.parse_args()
    directory = ROOT / args.directory
    directory.mkdir(parents=True, exist_ok=True)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    paths = list((ROOT/'client').glob('*.py')) + list((ROOT/'crates/imajev-runtime/src').rglob('*.rs')) + list((ROOT/'canisters/inference/src').rglob('*.rs'))
    paths += [ROOT/p for p in ['scripts/run_prefix_canister.py','scripts/run_full_canister.py','scripts/prepare_prefix_reuse.py','Cargo.lock','MODEL_LOCK.json','checkpoints/full-int8.manifest.json']]
    paths += list((ROOT/'crates/imajev-client/src').rglob('*.rs'))
    paths += [ROOT/'crates/imajev-client/Cargo.toml', ROOT/'canisters/inference/inference.did']
    paths.append(pathlib.Path(__file__))
    sources = {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
    wasm_hash = sha((ROOT/args.wasm).read_bytes())
    base = ['--canister',args.canister,'--wasm',args.wasm,
            '--reference','artifacts/reference-serving.json','--arithmetic','int8',
            '--wire-codec','bf16-block256-exact-v1','--frame-checksum','host' if args.host_checksum else 'blake3',
            '--row-cap','16384','--token-cap','132','--work-cap','2500000000',
            '--delta-head-cap','16','--attention-head-cap','8','--mlp-full-token-cap','89']
    flags = ['--compact-lossless','--compact-heads','--fuse-add-norm','--fuse-mlp',
             '--fuse-mlp-norm','--fuse-norm-rope','--fuse-delta','--wide-mlp',
             '--fuse-delta-input','--reuse-projection-inputs','--fuse-attention',
             '--fuse-delta-projected','--fuse-delta-finish','--fuse-mlp-pipeline',
             '--fuse-attention-full','--fuse-mlp-full','--fuse-delta-full-log',
             '--terminal-readout']
    if args.terminal_attention: flags.append('--fuse-terminal-attention')
    if args.terminal_decision:
        assert args.terminal_attention
        flags.append('--fuse-terminal-decision')
    if args.terminal_tail:
        assert args.terminal_decision and args.terminal_attention
        flags.append('--fuse-terminal-tail')
    def run(name, script, extra, common=True):
        cmd = [sys.executable,str(ROOT/script),*(base+flags if common else []),*extra]
        with (directory/f'{name}.log').open('w') as log:
            subprocess.run(cmd,check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(completed=name)),flush=True)
    def compare_states(new, old):
        for layer in range(32):
            name=f'layer-{layer:02d}'
            if layer==30 and args.terminal_tail and not (new/'queries'/f'{name}.npy').exists():
                layers=json.loads((new/'queries/layers.json').read_bytes())
                assert layers[30]['hidden_exported'] is False and layers[31]['included_in_layer']==30
            else:
                a=np.load(new/'queries'/f'{name}.npy',allow_pickle=False)
                b=np.load(old/'queries'/f'{name}.npy',allow_pickle=False)
                assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(), ('hidden',layer)
            with np.load(new/'queries/states'/f'{name}.npz',allow_pickle=False) as a, np.load(old/'queries/states'/f'{name}.npz',allow_pickle=False) as b:
                assert set(a.files)==set(b.files), ('state fields',layer)
                for key in a.files:
                    assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(), ('state',layer,key)
    results=[]
    def verify(label, baseline_label, decision=True):
        new=directory/label;old=ROOT/f'artifacts/output-pairs-v2-{baseline_label}'
        report=json.loads((new/'report.json').read_bytes());before=json.loads((old/'report.json').read_bytes())
        assert report['model']==before['model'] and report['pack_hash']==before['pack_hash'] and report['input_hash']==before['input_hash']
        assert report['wasm_sha256']==wasm_hash and report['replayed_queries']==0
        assert report['max_query_instructions']<=5_000_000_000
        failures=new/'queries/failures.jsonl'
        assert not failures.exists() or not failures.read_bytes()
        compare_states(new,old)
        if decision:
            assert report['comparison']['typed_output_valid']
            a=report['decision_query']['ok']['decision'];b=before['decision_query']['ok']['decision']
            for key in ['value','abstained','raw_logits','probabilities','unknown_probability']:
                assert a[key]==b[key], ('decision',key)
            assert np.load(new/'final-hidden.npy',allow_pickle=False).tobytes()==np.load(old/'final-hidden.npy',allow_pickle=False).tobytes()
        results.append(dict(label=label,baseline=baseline_label,full_bitwise_equal=True,hidden_scope='All exported hidden; compact terminal tail does not export layer30 hidden',query_count=report['query_count'],total_instructions=report['total_instructions'],total_candid_bytes=report['total_candid_bytes'],max_query_instructions=report['max_query_instructions'],max_observed_heap_bytes=report['max_observed_heap_bytes'],wall_seconds=report['wall_seconds_this_run'],baseline_report_sha256=sha((old/'report.json').read_bytes()),report_sha256=sha((new/'report.json').read_bytes())))
    prefix=directory/'prefix';cache=prefix/'queries';packets=directory/'packets'
    run('prefix','scripts/run_prefix_canister.py',['--record','0','--prepare-prefix','--prefix-tokens','45','--cache',str(cache),'--directory',str(prefix)])
    verify('prefix','prefix',False)
    run('codec','scripts/prepare_prefix_reuse.py',['--prefix-directory',str(prefix),'--directory',str(packets),'--canister',args.codec_canister,'--run-report',str(directory/'codec-preparation.json')],False)
    # Same candidate without hybrid packets isolates codec savings from compiler
    # and other already validated exact changes in this experimental build.
    run('baseline-617','scripts/run_prefix_canister.py',['--record','0','--cache',str(cache),'--directory',str(directory/'baseline-617')])
    verify('baseline-617','617')
    for label,record in [('617',0),('insufficient',19),('maximum',11)]:
        run(label,'scripts/run_prefix_canister.py',['--record',str(record),'--cache',str(cache),'--hybrid-cache',str(packets),'--directory',str(directory/label),*(['--fuse-prefix-start'] if args.prefix_start else [])])
        verify(label,label)
    run('normal','scripts/run_full_canister.py',['--record','0','--directory',str(directory/'normal')])
    verify('normal','normal')
    assert sources=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
    result=dict(scope='Full query inference; prefix/codec/model preparation reported separately; no mainnet deployment',canister=args.canister,wasm_sha256=wasm_hash,source_hashes=sources,cases=results,codec_preparation=json.loads((directory/'codec-preparation.json').read_bytes()),goal_50_verified=next(c for c in results if c['label']=='617')['query_count']<=50)
    (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
