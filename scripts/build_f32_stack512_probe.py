#!/usr/bin/env python3
"""Test output512 stack accumulation under the IC per-function locals limit."""
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'scripts/build_f32_output128_probe.py'
    source = p.read_text().replace('artifacts/f32-output128-v1/build', 'artifacts/f32-stack512-v1/build')
    for old, new in [('rows%128==0', 'rows%512==0'), ('step_by(128)', 'step_by(512)'),
                     ('[[0f32;128]', '[[0f32;512]'), ('x.as_ptr(),128,', 'x.as_ptr(),512,'),
                     ('r+128]', 'r+512]'), ('n*128', 'n*512')]:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = dict(__file__=__file__, __name__='f32_stack512_builder')
    exec(compile(source, str(p), 'exec'), namespace)
    stack = ROOT / 'scripts/build_f32_stack_probe.py'
    stack_source = stack.read_text().replace('width = 128', 'width = 512')
    kernel_ns = dict(__file__=__file__, __name__='f32_stack512_kernel')
    exec(compile(stack_source, str(stack), 'exec'), kernel_ns)
    wat = kernel_ns['kernel']()
    locals_count = len(re.findall(r'\(local \$', wat))
    assert locals_count + 9 <= 10_000, locals_count
    namespace['kernel'] = kernel_ns['kernel']
    namespace['main']()
    d = ROOT / 'artifacts/f32-stack512-v1'
    (d / 'frozen-builder.py').write_text(source)
    (d / 'frozen-kernel-generator.py').write_text(stack_source)
    (d / 'upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '\n')
    print(f'output512 kernel has {locals_count} locals; ascending F32 order unchanged')


if __name__ == '__main__':
    main()
