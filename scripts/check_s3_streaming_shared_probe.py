#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s3-k2-batch48-shared-probe-v1/frozen-checker.py'
 d=ROOT/'artifacts/s3-streaming-shared-probe-v1'
 source=original.read_text().replace('s3-k2-batch48-shared-probe-v1','s3-streaming-shared-probe-v1').replace('rank343 batch48/output128','rank343 shared streaming batch88/output256')
 frozen=d/'frozen-checker.py';assert not frozen.exists();frozen.write_text(source)
 (d/'checker-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
