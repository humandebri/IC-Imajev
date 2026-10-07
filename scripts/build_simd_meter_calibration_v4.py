#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/simd-meter-calibration-v3/frozen-builder.py';d=ROOT/'artifacts/simd-meter-calibration-v4';d.mkdir(exist_ok=False)
 source=original.read_text().replace('simd-meter-calibration-v3','simd-meter-calibration-v4')
 source=source.replace('(export "canister_update {name}")','(export "canister_update {name}_update")')
 source=source.replace("f'{name}:()->(nat64);'","f'{name}_update:()->(nat64);'")
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
