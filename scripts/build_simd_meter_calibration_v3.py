#!/usr/bin/env python3
"""Matched packet/splat bodies, multiple loads, and dual query/update exports."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/simd-meter-calibration-v2/frozen-builder.py';d=ROOT/'artifacts/simd-meter-calibration-v3';d.mkdir(exist_ok=False)
 source=original.read_text().replace('simd-meter-calibration-v2','simd-meter-calibration-v3')
 old='';new=''
 for k in range(4):
  old+=f'local.get $ptr v128.load32_splat offset={k*4} local.tee $r local.get $b i32x4.dot_i16x8_s local.set $r\n'
  new+=('local.get $ptr v128.load local.tee $packet\n'if k==0 else 'local.get $packet\n')
  new+='v128.const i32x4 0 0 0 0\ni8x16.shuffle '+' '.join(map(str,list(range(k*4,k*4+4))*4))+'\nlocal.tee $r local.get $b i32x4.dot_i16x8_s local.set $r\n'
 extra={'quad_splats':old,'quad_packet':new,'load128_twice':'local.get $ptr v128.load local.set $r\n'*2,'load128_four':'local.get $ptr v128.load local.set $r\n'*4}
 anchor=" text='''(module";assert source.count(anchor)==1;source=source.replace(anchor,' operations.update('+repr(extra)+')\n'+anchor)
 source=source.replace('(local $a v128)(local $b v128)','(local $packet v128)(local $a v128)(local $b v128)')
 source=source.replace('(func(export "canister_query {name}")','(func(export "canister_query {name}")(export "canister_update {name}")')
 anchor=" files=[Path(__file__),p,d/'probe.wasm',d/'probe.did']"
 extra_did=" (d/'probe-update.did').write_text('service:{'+''.join(f'{name}:()->(nat64);'for name in operations)+'}')\n"
 assert source.count(anchor)==1;source=source.replace(anchor,extra_did+anchor.replace("d/'probe.did']","d/'probe.did',d/'probe-update.did']"))
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
