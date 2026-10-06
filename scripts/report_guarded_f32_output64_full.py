#!/usr/bin/env python3
"""Independently check archived build, ordinary replies and full-model replay."""
import hashlib
import json
import ast
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/guarded-f32-output64-v1'
HELPER = ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(values):
    for path, expected in values.items():
        assert sha(ROOT/path) == expected, path


def main():
    build = json.loads((D/'full-build/report.json').read_text())
    kernel = json.loads((ROOT/'artifacts/delta-register-v3/check/report.json').read_text())
    full = json.loads((D/'full-proof-v1/report.json').read_text())
    identities(build['source_hashes']); identities(build['dependency_hashes'])
    identities(kernel['source_hashes']); identities(full['source_hashes'])
    assert build['wasm_sha256']==sha(D/'full-build/full.wasm')==full['candidate']
    assert all(p['wasmparser_validation'] for p in build['patches'])
    helper=ROOT/'artifacts/delta-writeback-target/debug/writeback_args'
    kernel_summaries=[]
    for case in kernel['cases']:
        assert case['bitwise_equal']
        for key in ['baseline','register']:
            m=case['measurements'][key];reply=ROOT/m['reply'];assert sha(reply)==m['reply_sha256']
            raw=json.loads(subprocess.check_output([str(helper),'decode',str(reply)],text=True))
            assert all(m[k]==v for k,v in raw.items())
            assert bytes(raw['digest']).hex()==case['native_digest']
        before=case['measurements']['baseline']['kernel_instructions'];after=case['measurements']['register']['kernel_instructions']
        assert abs(100*(1-after/before)-case['reduction_percent'])<1e-10
        kernel_summaries.append(dict(tokens=case['tokens'],reduction_percent=case['reduction_percent']))
    assert len(kernel_summaries)==7 and kernel['ordinary_queries']==14
    assert full['complete'] and full['baseline_snapshot_restored'] and full['snapshot_deleted']
    before = json.loads((D/'full-proof-v1/before.json').read_text())
    restored = json.loads((D/'full-proof-v1/restored.json').read_text())
    assert restored['restored'] and restored['module'] == before['module'] == full['baseline']
    assert restored['cache_equal'] and restored['pack_equal']
    assert full['preparation']['final'] == before['cache']
    prep_directory=D/'full-proof-v1/fixed-prefix-preparation'
    prep=json.loads((prep_directory/'report.json').read_text())
    identities(prep['source_hashes'])
    assert prep==full['fixed_prefix_preparation'] and prep['update_calls']==24 and prep['state_bytes']==50_331_648
    for number,row in enumerate(prep['layers'],1):
        layer=row['layer']
        assert sha(prep_directory/f'layer-{layer:02d}.log.f32')==row['log_sha256']
        assert sha(prep_directory/f'{layer:02d}.args.bin')==row['args_sha256']
        reply=prep_directory/f'{layer:02d}.reply.hex'
        assert sha(reply)==row['reply_sha256']
        measured=json.loads(subprocess.check_output([str(ROOT/'artifacts/prefix-state-cache-v1/prefix-state-args'),'decode',str(reply)],text=True))
        assert measured==row['result'] and measured['count']==number and measured['bytes']==number*2_097_152
    assert sum(r['result']['instructions'] for r in prep['layers'])==prep['instructions']
    summaries = []
    for case in full['cases']:
        proposal = case['proposal']
        directory = D/'full-proof-v1'/proposal
        assert sha(directory/'replay.json') == case['queries_report_sha256']
        original_report = ROOT/f'artifacts/boomdao-current-v1/{proposal}-r1/report.json'
        assert sha(original_report) == case['source_report_sha256']
        rows = json.loads((directory/'replay.json').read_text())
        assert len(rows) == case['queries'] == 39
        old_report = json.loads(original_report.read_text())
        for row in rows:
            i = row['index']
            assert row['bitwise_equal']
            assert sha(ROOT/row['source_request']) == sha(directory/f'{i:06d}.request.bin') == row['request_sha256']
            assert sha(ROOT/row['source_response']) == sha(directory/f'{i:06d}.response.bin') == row['response_sha256']
            assert 0 < row['candidate']['instructions'] < 5_000_000_000
            assert row['candidate']['request_bytes'] == row['baseline']['request_bytes']
            assert row['candidate']['reply_bytes'] == row['baseline']['reply_bytes']
            old_metric = old_report['queries'][i]
            assert old_metric['index']==i and old_metric['ok']==row['baseline']
            if row['op']=='mlp_delta_front':
                assert row['output_hashes'] == old_metric['outputs_sha256']
                for name,digest in row['output_hashes'].items():
                    assert sha(directory/name) == sha(original_report.parent/'queries'/name) == digest
                for suffix in ['prefix.bin','expected.bin']:
                    item = row['extras'][suffix]
                    assert sha(ROOT/item['source']) == sha(ROOT/item['copy']) == item['sha256']
                assert row['extras']['prefix_tokens'] == old_report['tokens']-old_report['processed_tokens']
            else:
                assert not row['extras'] and not row['output_hashes']
        baseline = sum(r['baseline']['instructions'] for r in rows)
        candidate = sum(r['candidate']['instructions'] for r in rows)
        assert (baseline,candidate) == (case['baseline_instructions'],case['candidate_instructions'])
        assert abs(100*(1-candidate/baseline)-case['reduction_percent']) < 1e-10
        assert case['decision']==rows[-1]['candidate']['decision']
        assert {k:v for k,v in case['decision'].items() if k!='instructions'}=={k:v for k,v in rows[-1]['baseline']['decision'].items() if k!='instructions'}
        assert max(r['candidate']['heap_pages'] for r in rows)*65536<2**32
        summaries.append({k:case[k] for k in ['proposal','suffix_tokens','queries','baseline_instructions','candidate_instructions','reduction_percent','max_query_instructions','candid_bytes']})
    assert len(summaries)==3 and full['ordinary_queries']==117
    report = dict(source_sha256=sha(Path(__file__)),kernel_report_sha256=sha(ROOT/'artifacts/delta-register-v3/check/report.json'),
        full_report_sha256=sha(D/'full-proof-v1/report.json'),baseline=full['baseline'],candidate=full['candidate'],
        kernel_cases=7,kernel_queries=14,kernel_summaries=kernel_summaries,full_queries=117,full_cases=summaries,
        all_frames_and_decision_values_bitwise_equal=True,decision_counter_excluded=True,baseline_module_and_cache_restored=True,
        scope='Counter reduction in fixed 39-query suffix graphs. Prefix/cache preparation excluded. No claim of fewer queries or broader accuracy improvement.')
    (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
