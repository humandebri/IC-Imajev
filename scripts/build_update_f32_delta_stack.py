#!/usr/bin/env python3
"""Combine verified stack-held F32 projection and Delta recurrence candidates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT / 'artifacts/update-f32-stack-v1'
    workflow = json.loads((upstream / 'workflow-hashes.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p,h in workflow.items())
    delta = json.loads((ROOT / 'artifacts/delta-stack-v1/summary.json').read_text())
    assert delta['native_output_and_final_state_bits_equal'] and delta['saved_replies_verified']
    assert delta['ordinary_queries'] == 14
    d = ROOT / 'artifacts/update-f32-delta-stack-v1'
    d.mkdir(exist_ok=False)
    for name in ['wide64.wat','wide128.wat']:
        (d / name).write_bytes((upstream / name).read_bytes())
    p = ROOT / 'artifacts/delta-stack-v1/build/kernel.wat'
    (d / 'delta.wat').write_text(p.read_text().replace('__imajev_delta_stack', '__imajev_delta_register'))
    source = (upstream / 'frozen-builder.py').read_text().replace('artifacts/update-f32-stack-v1', 'artifacts/update-f32-delta-stack-v1')
    anchor = "        assert sha(wat) == patch['source_sha256']"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + "\n        if i == 4:\n            wat = ROOT / 'artifacts/update-f32-delta-stack-v1/delta.wat'")
    (d / 'frozen-builder.py').write_text(source)
    paths = [Path(__file__), upstream / 'workflow-hashes.json', upstream / 'frozen-builder.py',
             ROOT / 'artifacts/delta-stack-v1/summary.json', p, d / 'frozen-builder.py', d / 'delta.wat', d / 'wide64.wat', d / 'wide128.wat']
    (d / 'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + '\n')
    exec(compile(source, str(upstream / 'frozen-builder.py'), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
