#!/usr/bin/env python3
"""Restore this benchmark's completed SSD staging directories to its artifact folder."""
import datetime
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/proposal-assessment-canister-20261005'
r = json.loads((D / 'report.json').read_text())
assert r['complete'] and r['completed_canister_inferences'] == r['runnable_unique_tasks']
metadata = json.loads((D / 'ssd-staging.json').read_text())
staging = Path(metadata['directory'])
assert staging.parent == Path('/private/tmp') and staging.name.startswith('imajev-proposal-query-staging-')

def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}

for name, target_name in metadata['links'].items():
    link, target = Path(name), Path(target_name)
    assert link.is_relative_to(D / 'runs') and target.is_relative_to(staging)
    assert link.is_symlink() and link.resolve() == target
    temporary = link.with_name(link.name + '.restoring')
    assert not temporary.exists()
    shutil.copytree(target, temporary)
    assert hashes(temporary) == hashes(target), 'staging copy differs'
    link.unlink()
    temporary.rename(link)
    shutil.rmtree(target)
assert not list(staging.iterdir())
staging.rmdir()
metadata['restored'] = True
metadata['restored_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(D / 'ssd-staging.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('restored', len(metadata['links']), 'directories; temporary SSD staging removed')
