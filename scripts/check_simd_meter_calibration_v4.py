#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/check_simd_meter_calibration_v3.py';d=ROOT/'artifacts/simd-meter-calibration-v4'
 source=original.read_text().replace('simd-meter-calibration-v3','simd-meter-calibration-v4')
 before="cmd=['icp','canister','call',CANISTER,name,'()'"
 assert source.count(before)==1;source=source.replace(before,"cmd=['icp','canister','call',CANISTER,(name+'_update'if mode=='update'else name),'()'")
 frozen=d/'frozen-checker.py';assert not frozen.exists();frozen.write_text(source)
 (d/'checker-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
