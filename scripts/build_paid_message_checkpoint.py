#!/usr/bin/env python3
"""Recompile the validated paid candidate with an absolute message checkpoint."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def leb(n):
    out = bytearray()
    while True:
        out.append((n & 127) | (128 if n > 127 else 0))
        n >>= 7
        if not n:
            return bytes(out)


def take(b, p):
    n = shift = 0
    while True:
        v = b[p]; p += 1
        n |= (v & 127) << shift
        if v < 128:
            return n, p
        shift += 7
        assert shift <= 35


def sections(b):
    assert b[:8] == b'\0asm\1\0\0\0'
    p = 8
    while p < len(b):
        kind = b[p]; size, p = take(b, p + 1)
        yield kind, b[p:p+size]
        p += size
    assert p == len(b)


def projection_bodies(parent, entries):
    bodies = []; exports = {}
    for kind, b in sections(parent):
        if kind == 7:
            n, p = take(b, 0)
            for _ in range(n):
                size, p = take(b, p); name = b[p:p+size].decode(); p += size
                k = b[p]; index, p = take(b, p + 1)
                if k == 0:
                    exports[name] = index
            assert p == len(b)
        elif kind == 10:
            n, p = take(b, 0)
            for _ in range(n):
                size, p = take(b, p); bodies.append(b[p:p+size]); p += size
            assert p == len(b)
    for entry in entries:
        assert exports[entry['export']] == entry['function_index']
        body = bodies[entry['code_position']]
        assert hashlib.sha256(body).hexdigest() == entry['replacement_body_sha256']
        # A single-function donor module for the existing strict typed patcher.
        # The final patched full module is validated by wasmparser.
        export = entry['export'].encode()
        parts = [(1, b'\1\x60\11' + b'\x7f'*9 + b'\0'), (3, b'\1\0'), (5, b'\1\0\1'), (7, b'\1' + leb(len(export)) + export + b'\0\0'), (10, b'\1' + leb(len(body)) + body)]
        yield b'\0asm\1\0\0\0' + b''.join(bytes([kind]) + leb(len(payload)) + payload for kind, payload in parts)


def main():
    parent = ROOT / 'artifacts/paid-stack-carry-projection-v1/build'
    base = json.loads((parent / 'report.json').read_text())
    assert sha(parent / 'full.wasm') == base['wasm_sha256']
    for key in ['source_hashes', 'dependency_hashes']:
        for p, h in base[key].items():
            assert sha(ROOT / p) == h, p
    d = ROOT / 'artifacts/paid-message-checkpoint-v1/build'
    d.mkdir(parents=True, exist_ok=False)
    for p in parent.glob('*.rs'):
        shutil.copyfile(p, d / p.name)
    old = (d / 'paid_inference.rs').read_text()
    new = (ROOT / 'canisters/inference/src/paid_inference.rs').read_text()
    assert old.count(' let started=ic_cdk::api::performance_counter(0);') == 1
    restored = new.replace(' // Counter zero starts at the IC message entry, before CDK argument decoding.\n // Do not subtract a checkpoint taken inside this function: that omits the\n // wrapper/prologue. This remains a checkpoint; the metric update and reply\n // epilogue below still need a separate whole-message accounting bound.\n let measured=ic_cdk::api::performance_counter(0);', ' let measured=ic_cdk::api::performance_counter(0)-started;')
    restored = restored.replace('fn inference_step(job_id:u64,expected_stage:u64)->Result<StepResult,String> {\n', 'fn inference_step(job_id:u64,expected_stage:u64)->Result<StepResult,String> {\n let started=ic_cdk::api::performance_counter(0);\n')
    assert restored == old, 'Only the reviewed checkpoint change is authorized here'
    (d / 'paid_inference.rs').write_text(new)
    command = base['command'][:]
    command[command.index('--edition=2021') + 1] = str(d / 'lib.rs')
    command[command.index('-o') + 1] = str(d / 'raw.wasm')
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(d), CARGO_PKG_NAME='imajev-inference', CARGO_PKG_VERSION='0.1.0', CARGO_PKG_VERSION_MAJOR='0', CARGO_PKG_VERSION_MINOR='1', CARGO_PKG_VERSION_PATCH='0', CARGO_PKG_VERSION_PRE='', CARGO_CRATE_NAME='imajev_inference')
    with (d / 'compiler.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=log)
    previous = d / 'raw.wasm'
    patches = []
    donors = projection_bodies((parent / 'full.wasm').read_bytes(), base['patches'])
    for i, (entry, donor) in enumerate(zip(base['patches'], donors, strict=True)):
        wat = d / f'projection-donor-{i}.wasm'
        wat.write_bytes(donor)
        target = d / ('full.wasm' if i == len(base['patches'])-1 else f'patched{i}.wasm')
        row = json.loads(subprocess.check_output([str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'), str(previous), str(wat), str(target), entry['export']], text=True))
        assert row['wasmparser_validation'] and row['replacement_body_sha256'] == entry['replacement_body_sha256']
        patches.append(row)
        previous = target
    assert len(patches) == 34
    for p in parent.glob('*.rs'):
        if p.name != 'paid_inference.rs':
            assert (d / p.name).read_bytes() == p.read_bytes()
    files = [Path(__file__), parent / 'report.json', parent / 'full.wasm', ROOT / 'canisters/inference/src/paid_inference.rs'] + list(d.glob('*.rs'))
    report = dict(complete=True, parent=base['wasm_sha256'], module=sha(d / 'full.wasm'), command=command, patches=patches, all34_projection_bodies_equal_to_validated_parent=True, source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files}, dependency_hashes=base['dependency_hashes'], scope='Absolute counter zero checkpoint includes the CDK prefix. Post-checkpoint metric updates and reply epilogue remain excluded. Build only; no installation or full paid proof. Historical validation pointers are unchanged.', full_paid_goal_achieved=False)
    (d / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(module=report['module'], patches=len(patches), full_paid_goal_achieved=False)), flush=True)


if __name__ == '__main__':
    main()
