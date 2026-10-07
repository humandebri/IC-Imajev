#!/usr/bin/env python3
"""Reuse immutable weights between prefix banks in the bulk-hash candidate."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    upstream=ROOT/'artifacts/update-prefix-hash-v1'
    hashes=json.loads((upstream/'workflow-hashes.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    diagnostic=ROOT/'artifacts/update-other-profile-v3/build'
    manifest=json.loads((diagnostic/'report.json').read_text())
    p=diagnostic/'update_inference.rs'
    assert sha(p)==manifest['source_hashes'][str(p.relative_to(ROOT))]
    text=p.read_text()
    reset=text[text.index('#[ic_cdk::update]\nfn reset_update_prefix()'):]
    assert reset.rstrip().endswith('}') and reset.count('fn reset_update_prefix')==1
    d=ROOT/'artifacts/update-prefix-hash-v2'
    d.mkdir(exist_ok=False)
    for p in list(upstream.glob('*.wat'))+[upstream/'prefix-helper.rs',upstream/'prefix-api.rs']:
        (d/p.name).write_bytes(p.read_bytes())
    (d/'reset-prefix.rs').write_text(reset)
    source=(upstream/'frozen-builder.py').read_text().replace('artifacts/update-prefix-hash-v1','artifacts/update-prefix-hash-v2')
    anchor="    runtime = base['runtime_command'][:]"
    assert source.count(anchor)==1
    addition="    p=D/'update_inference.rs'\n    p.write_text(p.read_text()+(D.parent/'reset-prefix.rs').read_text())\n"
    source=source.replace(anchor,addition+anchor)
    (d/'frozen-builder.py').write_text(source)
    files=[Path(__file__),upstream/'workflow-hashes.json',upstream/'frozen-builder.py',diagnostic/'report.json',diagnostic/'update_inference.rs',d/'reset-prefix.rs',d/'frozen-builder.py',d/'prefix-helper.rs',d/'prefix-api.rs']+list(d.glob('*.wat'))
    (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
    exec(compile(source,str(upstream/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
