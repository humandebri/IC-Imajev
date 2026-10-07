#!/usr/bin/env python3
"""Build WWC batch48/tile128 with chunked immutable prep and current control."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s3-k2-prepared-chunked-probe-v1/frozen-builder.py'
 d=ROOT/'artifacts/s3-k2-batch48-probe-v1';d.mkdir(exist_ok=False)
 source=original.read_text().replace('s3-k2-prepared-chunked-probe-v1','s3-k2-batch48-probe-v1').replace('s3-k2-prepared-kernels-v1','s3-k2-batch48-kernels-v1').replace('update-k2-odd-roots-v1','update-k2-pair-guard-fold-v1').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 before='for r in (0..rows).step_by(32){let wp=[self.data.as_ptr().add((r/32)*blocks*343*128)];'
 after='for r in (0..rows).step_by(128){let wp:[*const i16;4]=core::array::from_fn(|j|self.data.as_ptr().add((r/32+j)*blocks*343*128));'
 assert source.count(before)==1;source=source.replace(before,after)
 source=source.replace('rows%32!=0||rows>self.rows','rows%128!=0||rows>self.rows')
 source=source.replace('tile32','tile128').replace('n*32{{','n*128{{').replace('output_tile=32','output_tile=128')
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
