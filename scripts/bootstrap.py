#!/usr/bin/env python3
"""Download pinned official source only inside the project, then acquire immutable weights."""
import hashlib,pathlib,tarfile,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1];COMMIT='a0134749e0900189c129cd6bb5000969f3b64bb5';archive=ROOT/'artifacts/server.tar.gz';archive.parent.mkdir(exist_ok=True);(ROOT/'vendor').mkdir(exist_ok=True)
if not archive.exists():
 with urllib.request.urlopen(f'https://codeload.github.com/mohit67890/imajev/tar.gz/{COMMIT}')as r:archive.write_bytes(r.read())
with tarfile.open(archive)as t:t.extractall(ROOT/'vendor',filter='data')
print('official source archive sha256',hashlib.sha256(archive.read_bytes()).hexdigest())
