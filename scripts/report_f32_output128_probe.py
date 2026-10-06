#!/usr/bin/env python3
"""Recheck frozen query replies, input hashes and build identities before reporting."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/f32-output128-v1'
HELPER = ROOT / 'artifacts/f32-block-native/release/f32_args'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = D / 'build'
    b = json.loads((build/'report.json').read_text())
    for path, digest in b['source_hashes'].items():
        assert sha(ROOT/path) == digest, path
    assert sha(build/'diagnostic.wasm') == b['wasm_sha256']
    assert sha(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique') == b['patcher_sha256']
    assert all(p['wasmparser_validation'] for p in b['patches'])
    manifest = json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    assert manifest['model'] == sha(ROOT/'MODEL_LOCK.json')
    results = []
    for tensor in ['down-B']:
        directory = D/tensor
        r = json.loads((directory/'report.json').read_text())
        assert r['model'] == manifest['model']
        assert r['pack_hash'] == manifest['pack_hash']
        assert r['wasm_sha256'] == b['wasm_sha256']
        assert r['source_sha256'] == sha(ROOT/'scripts/check_f32_output128_probe.py')
        assert r['build_report_sha256'] == sha(build/'report.json')
        assert r['helper_sha256'] == sha(HELPER)
        weight = next(w for w in manifest['tensors'] if w['name'] == r['tensor'])
        with (ROOT/'checkpoints/full-int8.pack').open('rb') as stream:
            stream.seek(weight['offset'])
            weight_hash = hashlib.sha256(stream.read(weight['bytes'])).hexdigest()
        assert weight_hash == r['weight_sha256'] == sha(directory/'weights.bin')
        assert r['ordinary_queries'] == len(r['cases'])*2
        reductions = []
        for index, case in enumerate(r['cases']):
            path = directory/(case['label']+'.input.bin')
            assert sha(path) == case['input_sha256']
            assert path.stat().st_size == case['tokens']*case['cols']*4
            assert case['rows'] == weight['rows'] and case['cols'] == weight['cols']
            for method, key in [(0,'output32'), (1,'output128')]:
                count = r['preparation_updates'] + index*2 + method + 1
                reply = directory/f'{count:04d}-project.hex'
                measured = case['measurements'][key]
                assert sha(reply) == measured['reply_sha256']
                decoded = json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True))
                for field, value in decoded.items():
                    assert measured[field] == value, (tensor,case['label'],key,field)
                assert decoded['digest'] == case['native']['digest']
                assert decoded['output_values'] == case['tokens']*case['rows']
                assert decoded['total_instructions'] == decoded['input_prepare_instructions'] + decoded['project_instructions']
            before = case['measurements']['output32']['total_instructions']
            after = case['measurements']['output128']['total_instructions']
            reduction = 100*(1-after/before)
            assert abs(reduction+case['change_percent']) < 1e-10
            assert case['bitwise_equal']
            reductions.append(dict(label=case['label'],tokens=case['tokens'],
                                   before=before,after=after,reduction_percent=reduction))
        results.append(dict(tensor=tensor,report_sha256=sha(directory/'report.json'),
                            cases=len(r['cases']),queries=r['ordinary_queries'],
                            min_reduction_percent=min(c['reduction_percent'] for c in reductions),
                            max_reduction_percent=max(c['reduction_percent'] for c in reductions),
                            reductions=reductions))
    report = dict(wasm_sha256=b['wasm_sha256'],source_sha256=sha(pathlib.Path(__file__)),
                  queries=sum(r['queries'] for r in results),cases=sum(r['cases'] for r in results),
                  all_bits_equal_to_scalar_native=True,tensors=results,
                  scope='Isolated LoRA projection only. No full-model, accuracy-score or query-count claim.')
    (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
