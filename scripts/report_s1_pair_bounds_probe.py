#!/usr/bin/env python3
"""Independently check archived build, ordinary replies and full-model replay."""
import hashlib
import json
import ast
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/s1-pair-bounds-v1'
HELPER = ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(values):
    for path, expected in values.items():
        assert sha(ROOT/path) == expected, path


def main():
    build = json.loads((D/'build/report.json').read_text())
    kernel = json.loads((D/'check/report.json').read_text())
    full = json.loads((D/'full-proof-v2/report.json').read_text())
    patch = json.loads((D/'build/full.patch.json').read_text())
    identities(build['source_hashes']); identities(build['dependency_hashes'])
    identities(kernel['source_hashes']); identities(full['source_hashes'])
    assert build['wasm_sha256'] == sha(D/'build/diagnostic.wasm') == kernel['wasm_sha256']
    assert sha(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch') == build['patcher_sha256']
    assert all(p['wasmparser_validation'] for p in build['patches'])
    assert patch['wasmparser_validation'] and patch['replacement_body_sha256'] == build['patches'][-1]['replacement_body_sha256']
    assert patch['output_sha256'] == sha(D/'build/full.wasm') == full['candidate']
    assert patch['original_sha256'] == full['baseline'] == sha(ROOT/'artifacts/query-packing-v3/build/full.wasm')
    # Independently expand the retained tail outputs in the original variables.
    tree = ast.parse((ROOT/'scripts/generate_wat_s1.py').read_text())
    maps = {}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['A','B','C']:
            maps[node.targets[0].id] = ast.literal_eval(node.value)
    for output,terms in enumerate([{0:1,4:-1,6:1},{2:1,4:1}]):
        polynomial = {}
        for product,sign in terms.items():
            for x,xc in maps['A'][product].items():
                if x>=2: continue  # dummy second token is initialized zero
                for w,wc in maps['B'][product].items():
                    polynomial[x,w] = polynomial.get((x,w),0)+sign*xc*wc
        assert {k:v for k,v in polynomial.items() if v} == {(0,output):1,(1,2+output):1}
        assert sum(abs(sign)*sum(abs(v) for v in maps['A'][p].values())*127*sum(abs(v) for v in maps['B'][p].values())*128*128 for p,sign in terms.items()) < 2**31
    for n in range(1,513):
        t=2; fulln=n&-2
        while t<fulln:
            assert t+1<n
            t+=2
        if t<n: assert n>=3 and n%2==1 and t==n-1
    original = (ROOT/'artifacts/single-quad/build-v2/kernel3.wat').read_text()
    specialized = (D/'build/kernel.wat').read_text().replace('__imajev_s1_bounded_accumulate','__imajev_s1_wide_accumulate').replace(' (local $fulln i32)','')
    first_end = '(local.set $t (i32.const 2))'
    assert specialized.split(first_end)[0] == original.split(first_end)[0]
    manifest = json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    assert manifest['model'] == kernel['model'] == sha(ROOT/'MODEL_LOCK.json')
    weight = next(w for w in manifest['tensors'] if w['name'] == kernel['tensor'])
    with (ROOT/'checkpoints/full-int8.pack').open('rb') as f:
        f.seek(weight['offset']); data = f.read(weight['bytes'])
    assert hashlib.sha256(data).hexdigest() == kernel['weight_sha256']
    with zipfile.ZipFile(D/'build/source.zip') as archive:
        for name,digest in build['source_hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    kernel_summaries = []
    for case in kernel['cases']:
        assert case['bitwise_equal']
        ip = D/'check'/f'{case["label"]}.input.bin'
        assert sha(ip) == case['input_sha256']
        assert ip.stat().st_size == case['tokens']*case['cols']*4
        for key in ['original128','pair_bounds']:
            measured = case['measurements'][key]
            reply = ROOT/measured['reply']
            assert sha(reply) == measured['reply_sha256']
            decoded = json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True))
            assert all(measured[k] == v for k,v in decoded.items())
            assert decoded['digest'] == case['native']['digest']
            assert decoded['output_values'] == case['tokens']*case['rows']
            assert decoded['quantize_instructions']+decoded['input_prepare_instructions']+decoded['project_instructions'] == decoded['total_instructions']
        before = case['measurements']['original128']['total_instructions']
        after = case['measurements']['pair_bounds']['total_instructions']
        assert abs(100*(1-after/before)-case['reduction_percent']) < 1e-10
        kernel_summaries.append(dict(label=case['label'],tokens=case['tokens'],reduction_percent=case['reduction_percent']))
    assert len(kernel_summaries)==19 and kernel['ordinary_queries']==38
    assert full['complete'] and full['baseline_snapshot_restored'] and full['snapshot_deleted']
    before = json.loads((D/'full-proof-v2/before.json').read_text())
    restored = json.loads((D/'full-proof-v2/restored.json').read_text())
    assert restored['restored'] and restored['module'] == before['module'] == full['baseline']
    assert restored['cache_equal'] and restored['pack_equal']
    assert full['preparation']['final'] == before['cache']
    summaries = []
    for case in full['cases']:
        proposal = case['proposal']
        directory = D/'full-proof-v2'/proposal
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
        assert case['decision'] == rows[-1]['candidate']['decision'] == rows[-1]['baseline']['decision']
        summaries.append({k:case[k] for k in ['proposal','suffix_tokens','queries','baseline_instructions','candidate_instructions','reduction_percent','max_query_instructions','candid_bytes']})
    assert len(summaries)==3 and full['ordinary_queries']==117
    report = dict(source_sha256=sha(Path(__file__)),kernel_report_sha256=sha(D/'check/report.json'),
        full_report_sha256=sha(D/'full-proof-v2/report.json'),baseline=full['baseline'],candidate=full['candidate'],
        kernel_cases=19,kernel_queries=38,kernel_summaries=kernel_summaries,full_queries=117,full_cases=summaries,
        all_frames_and_decisions_bitwise_equal=True,baseline_module_and_cache_restored=True,
        symbolic_tail_identity=True,checked_integer_bound=True,first_pair_body_unchanged=True,
        scope='Counter reduction in fixed 39-query suffix graphs. Prefix/cache preparation excluded. No claim of fewer queries or broader accuracy improvement.')
    (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
