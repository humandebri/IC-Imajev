#!/usr/bin/env python3
"""Build current full K2 runtime with only specialized valid odd-row roots."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-k2-compact-v1';d=ROOT/'artifacts/update-k2-odd-roots-v1';k=ROOT/'artifacts/k2-odd-roots-kernels-v1';audit=json.loads((k/'integer-audit.json').read_text());assert audit['all_integer_dots_equal'];d.mkdir(exist_ok=False)
 for p,h in audit['source_hashes'].items():assert sha(ROOT/p)==h
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 for p in k.glob('*.wat'):(d/p.name).write_bytes(p.read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-k2-compact-v1','artifacts/update-k2-odd-roots-v1');(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),old/'build/report.json',old/'frozen-builder.py',k/'report.json',k/'integer-audit.json',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
