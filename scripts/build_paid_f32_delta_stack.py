#!/usr/bin/env python3
"""Build paid inference API with the F32/Delta stack candidate and unchanged billing."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/build_paid_update_full_candid.py'
    source = p.read_text().replace('artifacts/update-templates-v1/build', 'artifacts/update-f32-delta-stack-v1/build')
    source = source.replace('artifacts/paid-update-v1/build-v3', 'artifacts/paid-f32-delta-stack-v1/build')
    source = source.replace('artifacts/voting-template-prefix-v1/full-build/libimajev_runtime_register.rlib', 'artifacts/update-f32-delta-stack-v1/build/libimajev_runtime.rlib')
    source = source.replace("   if i==4:wat=ROOT/'artifacts/delta-register-v3/build/kernel.wat'", "   if i==4:wat=ROOT/'artifacts/update-f32-delta-stack-v1/delta.wat'")
    source = source.replace("   if i==5:wat=ROOT/'artifacts/f32-output64-v1/build-v2/wide.wat'", "   if i==5:wat=ROOT/'artifacts/update-f32-delta-stack-v1/wide64.wat'\n   if i==6:wat=ROOT/'artifacts/update-f32-delta-stack-v1/wide128.wat'")
    source = source.replace("  out=D/('full.wasm' if i==5", "  assert sha(wat)==patch['source_sha256']\n  out=D/('full.wasm' if i==6")
    source = source.replace('function_index\'] for p in patches})==6', 'function_index\'] for p in patches})==7')
    assert 'function_index\'] for p in patches})==7' in source
    d = ROOT / 'artifacts/paid-f32-delta-stack-v1'
    d.mkdir(exist_ok=False)
    (d / 'frozen-builder.py').write_text(source)
    upstream = ROOT / 'artifacts/update-f32-delta-stack-v1/workflow-hashes.json'
    hashes = json.loads(upstream.read_text())
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h for path,h in hashes.items())
    paths = [p, Path(__file__), upstream, d / 'frozen-builder.py']
    (d / 'workflow-hashes.json').write_text(json.dumps({str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, indent=2) + '\n')
    exec(compile(source, str(p), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
