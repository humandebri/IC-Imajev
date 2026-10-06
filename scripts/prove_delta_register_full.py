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
B = ROOT/'artifacts/delta-register-v3/full-build'
D = ROOT/'artifacts/delta-register-v3/full-proof-v1'
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
             ROOT/'artifacts/delta-register-v3/check/report.json']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    cases = []
    for proposal in ['617','620','653']:
        directory = ROOT/f'artifacts/boomdao-current-v1/{proposal}-r1'
        report = json.loads((directory/'report.json').read_text())
        assert report['wasm_sha256'] == BASELINE and report['replayed_queries'] == 0
        assert report['query_count'] == report['executed_query_count'] == len(report['queries']) == 39
        frozen = []
        output = D/proposal; output.mkdir()
        for metric in report['queries']:
            index = metric['index']
            request = directory/f'queries/{index:06d}.request.bin'
            response = directory/f'queries/{index:06d}.response.bin'
            target = output/f'{index:06d}.request.bin'; target.write_bytes(request.read_bytes())
            raw = target.read_bytes(); headlen = int.from_bytes(raw[:4],'little')
            header = json.loads(raw[4:4+headlen])
            extras = {}
            if metric['op']=='mlp_delta_front':
                assert header['op']=='mlp_stream_complete'
                assert set(metric['outputs_sha256']) == {f'{index:06d}.response.bin',f'{index:06d}.previous.bf16',f'{index:06d}.conv.bf16'}
                for suffix in ['prefix.bin','expected.bin']:
                    source = directory/f'queries/{index:06d}.{suffix}'
                    copy = output/f'{index:06d}.{suffix}'; copy.write_bytes(source.read_bytes())
                    extras[suffix] = dict(source=str(source.relative_to(ROOT)),sha256=sha(source),copy=str(copy.relative_to(ROOT)))
                expected = (output/f'{index:06d}.expected.bin').read_bytes()
                hlen = int.from_bytes(expected[:4],'little')
                expected_header = json.loads(expected[4:4+hlen])
                assert expected_header['op']=='mlp_stream_prepare'
                extras['front'] = expected_header['dims'][-1]
                extras['prefix_tokens'] = report['tokens']-report['processed_tokens']
                for name,digest in metric['outputs_sha256'].items():
                    assert sha(directory/'queries'/name)==digest
            else:
                assert header['op']==metric['op']
            frozen.append(dict(index=index,metric=metric,request=target,
                               source_request=str(request.relative_to(ROOT)),source_request_sha256=sha(request),
                               source_response=str(response.relative_to(ROOT)),source_response_sha256=sha(response),extras=extras))
        cases.append(dict(proposal=proposal,directory=directory,source_report_sha256=sha(directory/'report.json'),
                          baseline_report=report,requests=frozen,output=output))
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
        t = bridge(D/'replay')
        try:
            verify_module(t,candidate)
            for case in cases:
                rows = []
                for frozen in case['requests']:
                    index = frozen['index']; old = frozen['metric']
                    output = case['output']/f'{index:06d}.response.bin'
                    assert sha(frozen['request']) == frozen['source_request_sha256']
                    cmd = dict(op='step',input=str(frozen['request']),output=str(output))
                    if old['op']=='mlp_delta_front':
                        extras = frozen['extras']
                        cmd.update(op='mlp_delta_front',expected=str(ROOT/extras['expected.bin']['copy']),
                            prefix=str(ROOT/extras['prefix.bin']['copy']),prefix_tokens=extras['prefix_tokens'],front=extras['front'],
                            hidden=str(case['output']/f'{index:06d}.previous.bf16'),conv=str(case['output']/f'{index:06d}.conv.bf16'))
                    if 'decision_options' in old:
                        cmd.update(op='terminal_step_decision',options=old['decision_options'])
                    result = t.command(cmd)['ok']
                    assert result['instructions'] < 5_000_000_000
                    assert sha(output) == frozen['source_response_sha256'], (case['proposal'],index)
                    output_hashes = {}
                    if old['op']=='mlp_delta_front':
                        output_hashes = {name:sha(case['output']/name) for name in old['outputs_sha256']}
                        assert output_hashes == old['outputs_sha256']
                    assert (result['request_bytes'],result['reply_bytes']) == (old['ok']['request_bytes'],old['ok']['reply_bytes'])
                    if 'decision' in old['ok']:
                        assert result['decision'] == old['ok']['decision']
                    row = dict(index=index,op=old['op'],baseline=old['ok'],candidate=result,
                               request_sha256=sha(frozen['request']),response_sha256=sha(output),
                               source_request=frozen['source_request'],source_response=frozen['source_response'],
                               extras=frozen['extras'],output_hashes=output_hashes,bitwise_equal=True)
                    rows.append(row); write(case['output']/'replay.json',rows)
                baseline = case['baseline_report']['total_instructions']
                total = sum(r['candidate']['instructions'] for r in rows)
                assert baseline == sum(r['baseline']['instructions'] for r in rows)
                summary = dict(proposal=case['proposal'],tokens=case['baseline_report']['tokens'],
                    suffix_tokens=case['baseline_report']['processed_tokens'],queries=len(rows),
                    baseline_instructions=baseline,candidate_instructions=total,
                    reduction_percent=100*(1-total/baseline),max_query_instructions=max(r['candidate']['instructions'] for r in rows),
                    candid_bytes=sum(r['candidate']['request_bytes']+r['candidate']['reply_bytes'] for r in rows),
                    source_report_sha256=case['source_report_sha256'],all_response_frames_bitwise_equal=True,
                    decision=rows[-1]['candidate']['decision'],queries_report_sha256=sha(case['output']/'replay.json'))
                results.append(summary); write(D/'progress.json',results)
                print(json.dumps(summary),flush=True)
            verify_module(t,candidate)
            assert t.command(dict(op='weight_cache_status'))['ok']['cache'] == before_cache
        finally: t.close()
        assert sources == {str(p.relative_to(ROOT)):sha(p) for p in paths}
        for case in cases:
            assert sha(case['directory']/'report.json') == case['source_report_sha256']
            for frozen in case['requests']:
                assert sha(ROOT/frozen['source_request']) == frozen['source_request_sha256']
                assert sha(ROOT/frozen['source_response']) == frozen['source_response_sha256']
                for suffix in ['prefix.bin','expected.bin']:
                    if suffix in frozen['extras']:
                        item=frozen['extras'][suffix]
                        assert sha(ROOT/item['source']) == sha(ROOT/item['copy']) == item['sha256']
        write(D/'report.json',dict(complete=False,baseline=BASELINE,candidate=candidate,source_hashes=sources,
            cases=results,ordinary_queries=sum(r['queries'] for r in results),preparation=preparation,
            scope='Frozen request replay verifies all frame bytes and terminal decision. Fixed 39-query graph, not a new graph or query-count optimization.'))
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
    assert restored and len(results)==3
    report = json.loads((D/'report.json').read_text())
    report.update(complete=True,baseline_snapshot_restored=True,snapshot_deleted=True)
    write(D/'report.json',report)


if __name__ == '__main__':
    main()
