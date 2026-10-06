#!/usr/bin/env python3
"""Measure identical saved inference frames via query and replicated ingress.

No install/upgrade, weight preparation, graph rescheduling, or retry. Each reply
must match the frozen successful inference chain, including terminal decision.
Timing excludes client graph/codec work; cycle deltas include elapsed idle fees
and the status sampling calls and are not isolated inference charges.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--canister', default='6eydd-o3777-77775-aaama-cai')
    ap.add_argument('--proof', default='artifacts/single-quad/tail-proof-v2')
    ap.add_argument('--case', default='617')
    ap.add_argument('--directory', required=True)
    ap.add_argument('--count', type=int)
    a = ap.parse_args()
    source = ROOT / a.proof
    case = source / a.case
    dest = ROOT / a.directory
    dest.mkdir(parents=True, exist_ok=False)
    expected = json.loads((source / 'report.json').read_text())['module_sha256']
    requests = sorted((case / 'queries').glob('*.request.bin'))
    if a.count is not None:
        if not 1 <= a.count <= len(requests):
            raise ValueError('count bounds')
        requests = requests[:a.count]
    frozen = [source / 'report.json', case / 'report.json']
    records = []
    for req in requests:
        stem = req.name.removesuffix('.request.bin')
        response = req.with_name(stem + '.response.bin')
        metric = req.with_name(stem + '.metric.json')
        m = json.loads(metric.read_text())
        frozen += [req, response, metric]
        records.append((req, response, m))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in frozen}
    bridge = ROOT / 'target/release/imajev-client'
    process = subprocess.Popen([str(bridge), 'http://localhost:8001/', a.canister,
                                str(ROOT / 'artifacts/imajev-local.pem')],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def call(cmd):
        process.stdin.write(json.dumps(cmd) + '\n')
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('bridge exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result
    report = dict(scope='Frozen inference request chain executed again; not a fresh end-to-end graph run',
                  canister=a.canister, module_sha256=expected, case=a.case,
                  source_hashes=hashes, bridge_sha256=sha(bridge),
                  script_sha256=sha(pathlib.Path(__file__)), modes={})
    try:
        assert call(dict(op='module_hash'))['ok']['module_hash'] == expected, 'module mismatch'
        for mode in ['query', 'update']:
            directory = dest / mode
            directory.mkdir()
            before = call(dict(op='balance_status'))
            assert before['ok']['module_hash'] == expected
            rows = []
            started = time.perf_counter()
            for req, reference, metric in records:
                out = directory / reference.name
                cmd = dict(op='terminal_step_decision' if 'decision_options' in metric else 'step',
                           execution=mode, input=str(req), output=str(out))
                if 'decision_options' in metric:
                    cmd['options'] = metric['decision_options']
                result = call(cmd)
                assert out.read_bytes() == reference.read_bytes(), ('reply mismatch', mode, req.name)
                for key in ['instructions', 'request_bytes', 'reply_bytes', 'stable_read_bytes']:
                    assert result['ok'][key] == metric['ok'][key], (mode, req.name, key)
                if 'decision_options' in metric:
                    assert result['ok']['decision'] == metric['ok']['decision'], 'decision mismatch'
                row = dict(index=metric['index'], op=metric['op'], request_sha256=sha(req),
                           reply_sha256=sha(out), bitwise_equal=True, **result)
                rows.append(row)
                (directory / req.name.replace('.request.bin', '.metric.json')).write_text(json.dumps(row, indent=2)+'\n')
                print(json.dumps(dict(mode=mode, index=metric['index'], instructions=result['ok']['instructions'], wall_seconds=result['wall_seconds'])), flush=True)
            elapsed = time.perf_counter() - started
            after = call(dict(op='balance_status'))
            assert after['ok']['module_hash'] == expected
            result = dict(calls=len(rows), total_handler_instructions=sum(r['ok']['instructions'] for r in rows),
                          total_candid_bytes=sum(r['ok']['request_bytes']+r['ok']['reply_bytes'] for r in rows),
                          call_wall_seconds=sum(r['wall_seconds'] for r in rows), loop_wall_seconds=elapsed,
                          max_handler_instructions=max(r['ok']['instructions'] for r in rows),
                          all_reply_bytes_equal=True, terminal_decision_equal=any('decision' in r['ok'] for r in rows),
                          balance_before=before, balance_after=after,
                          observed_balance_decrease=int(before['ok']['cycles'])-int(after['ok']['cycles']),
                          cycle_scope='includes idle/storage fees and status calls; local subnet configuration',
                          rows=rows)
            report['modes'][mode] = result
            (dest / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        assert call(dict(op='module_hash'))['ok']['module_hash'] == expected
        assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in frozen}, 'reference changed'
        q, u = report['modes']['query'], report['modes']['update']
        report['comparison'] = dict(handler_instructions_equal=q['total_handler_instructions']==u['total_handler_instructions'],
                                    candid_bytes_equal=q['total_candid_bytes']==u['total_candid_bytes'],
                                    update_over_query_time=u['call_wall_seconds']/q['call_wall_seconds'])
        (dest / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:{key:value for key,value in v.items() if key!='rows'} for k,v in report['modes'].items()}, indent=2))
    except Exception as error:
        (dest / 'failure.json').write_text(json.dumps(dict(error=str(error), partial=report),indent=2)+'\n')
        raise
    finally:
        process.stdin.close()
        process.wait(timeout=15)

if __name__ == '__main__':
    main()
