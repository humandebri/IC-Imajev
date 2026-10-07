#!/usr/bin/env python3
"""Use the inference build's consistent dependency set for the conv diagnostic."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'artifacts/conv4-simd-v2/frozen-builder.py';s=p.read_text().replace('artifacts/conv4-simd-v2','artifacts/conv4-simd-v3')
 lo=s.index(' old=json.loads(');hi=s.index(" with (d/'compiler.log')",lo)
 s=s[:lo]+''' old=json.loads((ROOT/'artifacts/float-hash-v1/build/report.json').read_text())
 command=r['command'][:]
 command[command.index('--crate-name')+1]='imajev_conv4_probe'
 command[command.index('--edition=2021')+1]=str(d/'lib.rs')
 command[command.index('-o')+1]=str(d/'diagnostic.wasm')
 command=[v if not v.startswith('imajev_runtime=') else 'imajev_runtime='+str(d/'libimajev_runtime.rlib') for v in command]
'''+s[hi:]
 d=ROOT/'artifacts/conv4-simd-v3';d.mkdir(exist_ok=False);(d/'frozen-builder.py').write_text(s)
 files=[p,Path(__file__),ROOT/'artifacts/conv4-simd-v2/builder-hashes.json']
 (d/'builder-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
