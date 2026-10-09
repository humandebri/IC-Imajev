#!/usr/bin/env python3
"""Check anonymous access on an isolated, unprepared LOCAL public-query canister."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister', required=True)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    directory = (ROOT / args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    did = directory / 'query.did'
    did.write_text((ROOT / 'canisters/inference/paid-inference.did').read_text().replace(
        '  clearWeightCache', '  runAttentionInferenceStep : (blob, nat32) -> (variant { Err:text; Ok:reserved }) query;\n  clearWeightCache'))
    rows = []
    def call(method, arguments, query, allowed):
        command = ['icp', 'canister', 'call', args.canister, method, arguments,
                   '--network', 'local', '--identity', 'anonymous', '--candid', str(did)]
        if query:
            command.append('--query')
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
        output = result.stdout + result.stderr
        if allowed:
            assert result.returncode == 0 and 'owner only' not in output, (method, output)
        else:
            assert result.returncode != 0 and ('owner only' in output or 'has no update method' in output), (method, output)
        rows.append(dict(method=method, query=query, allowed=allowed, returncode=result.returncode,
                         stdout=result.stdout, stderr=result.stderr))
        (directory / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
    for method in ['getModelStatus', 'getWeightCacheStatus']:
        call(method, '()', True, True)
    for method, value in [('runInferenceStep','(blob "")'),
                          ('runFinalInferenceStep','(blob "", vec {"yes";"no"})'),
                          ('runDeltaInferenceStep','(blob "", blob "", 27:nat32, 256:nat32)'),
                          ('runAttentionInferenceStep','(blob "", 256:nat32)')]:
        call(method, value, True, True)  # Invalid input must return Err, not fail caller authorization.
        assert 'Err' in rows[-1]['stdout'], rows[-1]
    for method, value in [('clearWeightCache','()'), ('prepareModelUpload','("{}")'),
                          ('prepareWeightCache','("x")'),
                          ('verifyModelUpload','(1:nat64)'), ('installInferencePrefix','(0:nat32, vec {}, blob "")')]:
        call(method, value, False, False)
    call('runInferenceStep', '(blob "")', False, False)
    call('runPaidInferenceWorker', '(0:nat64, 0:nat64)', False, True)
    assert 'self only' in rows[-1]['stdout'], rows[-1]
    call('getInferenceReceipt', '("foreign-or-missing")', True, True)
    assert 'null' in rows[-1]['stdout'], rows[-1]
    call('retryInferenceRefund', '("foreign-or-missing")', False, True)
    assert 'missing receipt' in rows[-1]['stdout'], rows[-1]
    request = 'record {model="' + 'a'*64 + '";version=1:nat32;token_ids=vec {1:nat32};options=vec {"yes";"no"}}'
    call('runPaidInference', '(' + request + ', "anonymous-test")', False, True)
    assert 'caller/request ID' in rows[-1]['stdout'], rows[-1]
    (directory / 'report.json').write_text(json.dumps(dict(canister=args.canister,
        network='local', anonymous_checks=len(rows), checks=rows,
        scope='Access and malformed-input checks only. No model weights installed, no full inference claim.'), indent=2) + '\n')
    print(f'{len(rows)} anonymous local access checks passed')

if __name__ == '__main__':
    main()
