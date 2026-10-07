#!/usr/bin/env python3
"""Add empty-callee local-count calibration to matched SIMD instruction loops."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/build_simd_meter_calibration.py'
 d=ROOT/'artifacts/simd-meter-calibration-v2';d.mkdir(exist_ok=False)
 source=original.read_text().replace('simd-meter-calibration-v1','simd-meter-calibration-v2').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 anchor=" text='''(module"
 addition=" operations.update({'empty100':'','call_empty':'call $locals0','call_v128_64':'call $locals64','call_v128_5488':'call $locals5488','call_v128_8192':'call $locals8192','call_i32_8192':'call $intlocals8192'})\n"
 assert source.count(anchor)==1;source=source.replace(anchor,addition+anchor)
 source=source.replace('  text+=f\'\'\'(func',"  iterations=100 if name=='empty100' or name.startswith('call_') else 10000\n  text+=f'''(func")
 assert '(local.set $i(i32.const 10000))'in source
 source=source.replace('(local.set $i(i32.const 10000))','(local.set $i(i32.const {iterations}))')
 anchor=" text+=')\\n';p=d/'probe.wat'"
 helpers=" text+='(func $locals0)\\n'+''.join(f'(func $locals{n}'+''.join('(local v128)'for _ in range(n))+')\\n'for n in [64,5488,8192])+'(func $intlocals8192'+''.join('(local i32)'for _ in range(8192))+')\\n'\n"
 assert source.count(anchor)==1;source=source.replace(anchor,helpers+anchor)
 source=source.replace('iterations=10000,operations=operations,',"iterations=10000,method_iterations={name:(100 if name=='empty100' or name.startswith('call_') else 10000)for name in operations},operations=operations,")
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
