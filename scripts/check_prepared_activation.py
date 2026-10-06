#!/usr/bin/env python3
"""Same-Wasm exact finite BF16 lookup validation and isolated metering.

Representative inputs here are synthetic; this is not a full-model speed claim.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canister', required=True)
    ap.add_argument('--directory', default='artifacts/prepared-activation/check')
    a = ap.parse_args()
    d = ROOT / a.directory
    d.mkdir(parents=True, exist_ok=True)
    helper = ROOT / 'artifacts/prepared-activation/native/release/activation_args'
    wasm = ROOT / 'artifacts/prepared-activation/wasm/wasm32-unknown-unknown/release/imajev_activation_bench.wasm'
    sha = lambda b: hashlib.sha256(b).hexdigest()
    paths = sorted(set(list((ROOT / 'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml'] +
                       list((ROOT / 'scripts/activation_bench').rglob('*.rs')) +
                       [ROOT / 'crates/imajev-runtime/Cargo.toml', ROOT / 'Cargo.lock',
                        ROOT / 'scripts/activation_bench/Cargo.toml',
                        ROOT / 'scripts/activation_bench/Cargo.lock', pathlib.Path(__file__)]))
    hashes = lambda: {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}
    sources = hashes()
    common = ['--network', 'local', '--identity', 'imajev-local']

    def status():
        r = subprocess.check_output(['icp', 'canister', 'status', a.canister, *common, '--json'], cwd=ROOT, text=True)
        return json.loads(r)['module_hash'].removeprefix('0x')

    assert status() == sha(wasm.read_bytes())
    prep_hex = subprocess.check_output(['icp', 'canister', 'call', a.canister, 'prepare', *common,
                                        '()', '--output', 'hex'], cwd=ROOT, text=True)
    (d / 'preparation.hex').write_text(prep_hex)
    prep = json.loads(subprocess.check_output([str(helper), 'decode', str(d / 'preparation.hex'), 'preparation'], text=True))
    assert prep['bytes'] == 1_048_576
    codes = np.arange(65536, dtype=np.uint32)
    finite = codes[(codes & 0x7f80) != 0x7f80] << 16
    dense = np.linspace(-8, 8, 131072, dtype=np.float32).view(np.uint32)
    dense = (dense + 0x7fff + ((dense >> 16) & 1)) & 0xffff0000
    fallback = np.array([0x3f800001, 0xbf800001, 1, 0x80000001,
                         0x7f800000, 0xff800000, 0x7fc10000, 0xffc10000], dtype=np.uint32)
    cases = []
    for label, bits in [('all-finite-bf16', finite), ('synthetic-range-8', dense), ('fallback', fallback)]:
        inp = d / f'{label}.f32.bin'
        inp.write_bytes(bits.astype('<u4').tobytes())
        for kind, name in enumerate(['bf_sigmoid', 'bf_silu', 'bf_softplus', 'silu']):
            measures = {}
            for cached in [False, True]:
                arg = d / f'{label}-{kind}-{cached}.args.bin'
                subprocess.run([str(helper), 'query', str(inp), str(kind), str(cached).lower(), str(arg)], check=True)
                begin = time.monotonic()
                reply = subprocess.check_output(['icp', 'canister', 'call', a.canister, 'measure', *common,
                                                 '--query', '--args-file', str(arg), '--args-format', 'bin',
                                                 '--output', 'hex'], cwd=ROOT, text=True)
                elapsed = time.monotonic() - begin
                rp = d / f'{label}-{kind}-{cached}.hex'
                rp.write_text(reply)
                r = json.loads(subprocess.check_output([str(helper), 'decode', str(rp), 'measurement'], text=True))
                r.update(wall_seconds=elapsed, request_candid_bytes=arg.stat().st_size,
                         reply_hex_sha256=sha(reply.encode()))
                measures['lookup' if cached else 'original'] = r
            assert measures['lookup']['digest'] == measures['original']['digest'], (label, name)
            row = dict(input=label, function=name, input_sha256=sha(inp.read_bytes()),
                       values=len(bits), bitwise_equal=True, measurements=measures)
            cases.append(row)
            print(json.dumps(dict(input=label, function=name, bitwise_equal=True,
                                  original=measures['original']['kernel_instructions'],
                                  lookup=measures['lookup']['kernel_instructions'])), flush=True)
    assert status() == sha(wasm.read_bytes())
    assert hashes() == sources
    report = dict(scope='Isolated same-Wasm primitive validation. Synthetic range timing is not model timing. Handler counter excludes CDK decode, hash and reply serialization.',
                  canister=a.canister, wasm_sha256=sha(wasm.read_bytes()), helper_sha256=sha(helper.read_bytes()),
                  source_hashes=sources, preparation=prep, ordinary_queries=len(cases)*2,
                  certified_module_reads=2, cases=cases)
    (d / 'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
