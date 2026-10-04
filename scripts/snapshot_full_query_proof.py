#!/usr/bin/env python3
"""Create an immutable source snapshot from a completed full-query proof."""
import argparse
import hashlib
import json
import pathlib
import zipfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--directory',required=True)
args=ap.parse_args()
directory=(ROOT/args.directory).resolve()
report=json.loads((directory/'report.json').read_text())
contents={}
for name,expected in report['source_hashes'].items():
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT) or pathlib.Path(name).is_absolute():
        raise ValueError('source path outside repository')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError(f'current source does not match completed proof: {name}')
    contents[name]=data
# Exclusive creation: never replace previously recorded proof source.
with zipfile.ZipFile(directory/'validated-source.zip','x',zipfile.ZIP_DEFLATED) as archive:
    for name,data in contents.items():
        archive.writestr(name,data)
print(f'snapshotted {len(contents)} source files')
