#!/usr/bin/env python3
"""Measure saved MLP completion and next Delta as separate ordinary queries.

Their sum is only a scheduling estimate, not a fused-query instruction result.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from transport import Transport, decode
from prefix_inference import verify_module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--canister', required=True)
    ap.add_argument('--wasm', required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--layers', default='0,1,3')
    args = ap.parse_args()
    directory = ROOT / args.directory
    directory.mkdir(parents=True, exist_ok=True)
    mlp = ROOT / 'artifacts/mlp-pipeline-v1-617'
    delta = ROOT / 'artifacts/prefix_codec/full-owned-state-proof/baseline-617'
    sources = [(p, (p / 'report.json').read_bytes()) for p in [mlp, delta]]
    reports = [json.loads(data) for _, data in sources]
    for field in ['model', 'pack_hash', 'input_hash']:
        assert reports[0][field] == reports[1][field]
    sha = lambda b: hashlib.sha256(b).hexdigest()
    module = sha((ROOT / args.wasm).read_bytes())
    transport = Transport(reports[0]['model'], 'http://localhost:8001/', args.canister,
                          str(ROOT / 'artifacts/imajev-local.pem'), directory, reports[0]['pack_hash'])
    rows = []
    try:
        verify_module(transport, module)
        for layer in map(int, args.layers.split(',')):
            assert 0 <= layer < 31 and (layer + 1) % 4 != 3
            stages = []
            for index, (op, which) in enumerate([('mlp_down_norm_prepared', layer), ('delta_full_log_integer', layer+1)]):
                source, data = sources[index]
                query = next(q for q in reports[index]['queries'] if q['op'] == op
                             and int(re.search(r'\.layers\.(\d+)\.', q['tensor']).group(1)) == which)
                request = source / 'queries' / f'{query["index"]:06d}.request.bin'
                expected = source / 'queries' / f'{query["index"]:06d}.response.bin'
                output = directory / f'{layer:02d}-{index}.response.bin'
                result = transport.command(dict(op='step', input=str(request), output=str(output)))['ok']
                old_header, old_values = decode(expected.read_bytes())
                new_header, new_values = decode(output.read_bytes())
                assert old_header == new_header and old_values.tobytes() == new_values.tobytes()
                stages.append(dict(op=op, layer=which, request_sha256=sha(request.read_bytes()),
                                   expected_response_sha256=sha(expected.read_bytes()),
                                   source_report_sha256=sha(data), bitwise_equal=True, **result))
            total = sum(stage['instructions'] for stage in stages)
            rows.append(dict(layer=layer, stages=stages, instruction_sum=total,
                             estimated_headroom=5_000_000_000-total))
        verify_module(transport, module)
    finally:
        transport.close()
    report = dict(scope=__doc__, canister=args.canister, module_sha256=module,
                  script_sha256=sha(pathlib.Path(__file__).read_bytes()), cases=rows)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([(row['layer'], row['instruction_sum'], row['estimated_headroom']) for row in rows]))


if __name__ == '__main__':
    main()
