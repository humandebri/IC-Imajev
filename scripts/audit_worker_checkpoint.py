#!/usr/bin/env python3
"""Audit the actual IC checkpoint replies and donor/full-module preservation."""
import hashlib
import json
from pathlib import Path
from build_paid_message_checkpoint import sections, take

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_bodies(data):
    found = []
    for kind, b in sections(data):
        if kind == 10:
            count, p = take(b, 0)
            for _ in range(count):
                size, p = take(b, p)
                found.append(b[p:p+size]); p += size
            assert p == len(b)
    return found


def main():
    d = ROOT / 'artifacts/worker-wrapper-checkpoint-ic-v1'
    ic = json.loads((d / 'report.json').read_text())
    assert ic['complete'] and len(ic['replies']) == 9
    for p, h in ic['source_hashes'].items():
        assert sha(ROOT / p) == h, p
    counts = []
    for row in ic['replies']:
        raw = bytes.fromhex((ROOT / row['reply']).read_text().strip().removeprefix('0x'))
        if row['query']:
            assert len(raw) == 15 and raw[:7] == b'DIDL\0\1\x78'
            value = int.from_bytes(raw[7:], 'little')
            assert value == row['value']; counts.append(value)
        else:
            assert raw == b'DIDL\0\0'
    assert counts == [0, ic['first_counter'], ic['first_counter'], ic['subsequent_counter']]
    assert counts[3] > counts[1] > 0
    b = ROOT / 'artifacts/paid-wrapper-checkpoint-v1/build'
    build = json.loads((b / 'report.json').read_text())
    for p, h in build['source_hashes'].items():
        assert sha(ROOT / p) == h, p
    old = code_bodies((b / 'patched-33.wasm').read_bytes())
    new = code_bodies((b / 'full.wasm').read_bytes())
    wrapper = build['wrapper']
    imports = 0
    for kind, payload in sections((b / 'patched-33.wasm').read_bytes()):
        if kind == 2:
            count, at = take(payload, 0)
            for _ in range(count):
                for _ in range(2):
                    size, at = take(payload, at); at += size
                assert payload[at] == 0, 'This audit expects function-only imports'
                _, at = take(payload, at + 1); imports += 1
            assert at == len(payload)
    excluded = {wrapper['getter_function']-imports, wrapper['arm_function']-imports}
    assert len(new) == len(old)+1
    for i, body in enumerate(old):
        if i not in excluded:
            assert body == new[i], i
    assert new[-1] == bytes(wrapper['wrapper_body'])
    assert sha(b / 'full.wasm') == build['module']
    files = [Path(__file__), ROOT / 'scripts/build_paid_message_checkpoint.py', d / 'report.json', b / 'report.json', b / 'full.wasm', b / 'patched-33.wasm']
    result = dict(complete=True, raw_ic_replies_verified=9, original_worker_and_all_other_original_bodies_preserved=True, full_paid_candidate_verified=False, whole_message_upper_bound_proven=False, full_paid_goal_achieved=False, source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    (d / 'independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
