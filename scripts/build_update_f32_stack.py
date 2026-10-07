#!/usr/bin/env python3
"""Build unprofiled update inference with stack-held exact F32 sums in both tile widths."""
import hashlib
import json
from pathlib import Path
import build_f32_stack_probe as generator

ROOT = Path(__file__).resolve().parents[1]


def main():
    upstream = ROOT / 'scripts/build_update_f32_output128.py'
    source = upstream.read_text().replace('artifacts/update-f32-output128-v1/build', 'artifacts/update-f32-stack-v1/build')
    # Generate both widths from exactly the same ascending-column algorithm.
    k128 = generator.kernel()
    text = Path(generator.__file__).read_text().replace('width = 128', 'width = 64')
    namespace = dict(__file__=generator.__file__, __name__='stack64_generator')
    exec(compile(text, generator.__file__, 'exec'), namespace)
    k64 = namespace['kernel']()
    directory = ROOT / 'artifacts/update-f32-stack-v1'
    directory.mkdir(exist_ok=False)
    (directory / 'wide64.wat').write_text(k64)
    (directory / 'wide128.wat').write_text(k128.replace('__imajev_f32_wide"', '__imajev_f32_wide128"'))
    assert source.count("        assert sha(wat) == patch['source_sha256']") == 1
    source = source.replace("        assert sha(wat) == patch['source_sha256']", "        assert sha(wat) == patch['source_sha256']\n        if i == 5:\n            wat = ROOT / 'artifacts/update-f32-stack-v1/wide64.wat'")
    source = source.replace("wat.write_text(original.read_text().replace('__imajev_f32_wide\"', '__imajev_f32_wide128\"'))", "wat.write_text((ROOT / 'artifacts/update-f32-stack-v1/wide128.wat').read_text())")
    assert "wat.write_text((ROOT / 'artifacts/update-f32-stack-v1/wide128.wat')" in source
    (directory / 'frozen-builder.py').write_text(source)
    (directory / 'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [upstream, Path(__file__), Path(generator.__file__), directory / 'frozen-builder.py', directory / 'wide64.wat', directory / 'wide128.wat']}, indent=2) + '\n')
    exec(compile(source, str(upstream), 'exec'), dict(__file__=__file__, __name__='__main__'))


if __name__ == '__main__':
    main()
