#!/usr/bin/env python3
"""Package reviewed working sources and pinned browser assets without Git history."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from check_publication import ROOT, git, scan, working_files


def read_entry(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
        raise ValueError('Unsafe release path: ' + name)
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink in release path: ' + name)
    data = current.read_bytes()
    findings = scan(data, name)
    if findings:
        raise ValueError('Publication candidates: ' + json.dumps(findings))
    return data, 0o755 if current.stat().st_mode & 0o111 else 0o644


def write_archive(output, entries):
    # Fixed gzip/tar metadata makes repeated packages of identical inputs identical.
    with output.open('xb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for name, (data, mode) in sorted(entries.items()):
                entry = tarfile.TarInfo('ic-imajev/' + name)
                entry.size, entry.mode, entry.mtime = len(data), mode, 0
                archive.addfile(entry, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/public-release/ic-imajev-source.tar.gz')
    args = parser.parse_args()
    count, findings = working_files()
    if findings:
        raise SystemExit('Working-tree publication check failed: ' + json.dumps(findings))
    # An exact allow-list from the checksum verifier, never a recursive copy of ignored data.
    assets = json.loads(subprocess.check_output(['node', str(ROOT / 'frontend/scripts/check-release-assets.mjs'), '--files'], cwd=ROOT))
    paths = set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''}
    entries = {}
    for name in sorted(paths):
        if (ROOT / name).exists():
            entries[name] = read_entry(ROOT, name)
    for name in assets:
        entries[name] = read_entry(ROOT, name)
    license_data = read_entry(ROOT, 'checkpoints/base/LICENSE')
    lock = json.loads((ROOT / 'MODEL_LOCK.json').read_text())
    if hashlib.sha256(license_data[0]).hexdigest() != lock['components']['base']['files']['LICENSE']['sha256']:
        raise SystemExit('Pinned upstream license checksum mismatch')
    entries['docs/licenses/Qwen-Apache-2.0.txt'] = license_data
    inventory = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'mode': mode}
                 for name, (data, mode) in sorted(entries.items())}
    entries['PUBLIC_SOURCE_MANIFEST.json'] = ((json.dumps({'format': 1, 'scope': 'Working source and pinned browser assets; excludes Git history, model weights and frozen optimized build inputs.', 'files': inventory}, indent=2) + '\n').encode(), 0o644)
    output = args.output.resolve()
    relative = output.relative_to(ROOT)
    if subprocess.run(['git', 'check-ignore', '-q', str(relative)], cwd=ROOT).returncode:
        raise SystemExit('Output must be inside an ignored repository directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_archive(output, entries)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({'file': str(relative), 'sha256': digest, 'working_files_checked': count, 'archive_files': len(entries)}, indent=2))


if __name__ == '__main__':
    main()
