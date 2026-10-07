#!/usr/bin/env python3
"""Independently decode all raw Candid component replies and check their hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Reply:
    def __init__(self, data):
        self.data = data
        self.at = 0

    def take(self, n):
        v = self.data[self.at:self.at+n]
        assert len(v) == n
        self.at += n
        return v

    def leb(self, signed=False):
        n = shift = 0
        while True:
            v = self.take(1)[0]
            n |= (v & 127) << shift
            shift += 7
            assert shift <= 70
            if v < 128:
                return n - (1 << shift) if signed and v & 64 else n

    def decode(self):
        assert self.take(4) == b'DIDL'
        assert self.leb() == 3
        # Exact Result<(u64, Vec<String>), String> schema; reject other types.
        assert self.leb(True) == -21 and self.leb() == 2
        assert (self.leb(), self.leb(True)) == (17724, 1)  # Ok -> record
        assert (self.leb(), self.leb(True)) == (3456837, -15)  # Err -> text
        assert self.leb(True) == -20 and self.leb() == 2
        assert (self.leb(), self.leb(True)) == (0, -8)  # nat64
        assert (self.leb(), self.leb(True)) == (1, 2)
        assert self.leb(True) == -19 and self.leb(True) == -15  # vec text
        assert self.leb() == 1 and self.leb(True) == 0
        assert self.leb() == 0  # Ok
        counter = int.from_bytes(self.take(8), 'little')
        assert self.leb() == 4
        digests = [self.take(self.leb()).decode('ascii') for _ in range(4)]
        assert self.at == len(self.data)
        return counter, digests


def main():
    d = ROOT / 'artifacts/sha256-four-lane-meter-v1'
    report = json.loads((d / 'report.json').read_text())
    assert report['complete'] and len(report['replies']) == 18
    for p, h in report['source_hashes'].items():
        assert sha(ROOT / p) == h, p
    build = json.loads((d / 'build.json').read_text())
    for p, h in build['source_hashes'].items():
        assert sha(ROOT / p) == h, p
    assert sha(d / 'probe.wasm') == report['module'] == build['wasm_sha256']
    for row in report['replies']:
        p = ROOT / row['reply']
        count, hashes = Reply(bytes.fromhex(p.read_text().strip().removeprefix('0x'))).decode()
        expected = [sha(ROOT / m['path']) for m in build['groups'][row['group']]['messages']]
        assert count == row['instructions'] and hashes == row['digests'] == expected
    for c in report['comparisons']:
        rows = {r['simd']:r['instructions'] for r in report['replies'] if r['group'] == c['group']}
        assert c['scalar'] == rows[False] and c['simd'] == rows[True]
        assert c['saved'] == c['scalar'] - c['simd'] > 0
        assert c['percent_saved'] == (1-c['simd']/c['scalar'])*100
    before = json.loads((d / 'check/full-pre-status.json').read_text())
    after = json.loads((d / 'check/full-post-status.json').read_text())
    assert before['module_hash'] == after['module_hash'] and before['status'] == after['status'] == 'Running'
    own = json.loads((d / 'check/post-status.json').read_text())
    assert own['status'] == 'Stopped' and own['module_hash'].removeprefix('0x') == report['module']
    files = [Path(__file__), d / 'report.json', d / 'build.json'] + [ROOT / r['reply'] for r in report['replies']]
    result = dict(complete=True, independently_decoded_replies=18, independently_verified_ic_digests=72, component_savings_percent_range=[min(c['percent_saved'] for c in report['comparisons']),max(c['percent_saved'] for c in report['comparisons'])], whole_inference_savings_measured=False, full_paid_goal_achieved=False, source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    (d / 'independent-ic-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
