#!/usr/bin/env python3
"""Matched IC query loops to measure installed-runtime SIMD metering deltas."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/simd-meter-calibration-v1';d.mkdir(exist_ok=False)
 operations={
 'empty':'', 'i32_const':'i32.const 0 drop',
 'i32_get':'local.get $ptr drop', 'v128_const':'v128.const i32x4 0 0 0 0 drop',
 'v128_get':'local.get $a drop',
 'v128_set':'v128.const i32x4 0 0 0 0 local.set $r',
 'v128_tee':'local.get $a local.tee $r drop',
 'dot':'local.get $a local.get $b i32x4.dot_i16x8_s local.set $r',
 'i32x4_add':'local.get $a local.get $b i32x4.add local.set $r',
 'i16x8_add':'local.get $a local.get $b i16x8.add local.set $r',
 'i16x8_sub':'local.get $a local.get $b i16x8.sub local.set $r',
 'shuffle':'local.get $a local.get $b i8x16.shuffle 0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 local.set $r',
 'load32_splat':'local.get $ptr v128.load32_splat local.set $r',
 'load128':'local.get $ptr v128.load local.set $r',
 'load8x8_s':'local.get $ptr v128.load8x8_s local.set $r',
 'f32x4_mul':'local.get $a local.get $b f32x4.mul local.set $r',
 'f32x4_add':'local.get $a local.get $b f32x4.add local.set $r',
 'f32x4_convert':'local.get $a f32x4.convert_i32x4_s local.set $r',
 'i32_mul':'local.get $ptr i32.const 2 i32.mul drop',
 'i32_shl':'local.get $ptr i32.const 1 i32.shl drop',
 }
 text='''(module
 (import "ic0" "performance_counter" (func $counter(param i32)(result i64)))
 (import "ic0" "msg_reply_data_append" (func $append(param i32 i32)))
 (import "ic0" "msg_reply" (func $reply))
 (memory(export "memory")1)
 (data(i32.const 0) "DIDL\\00\\01\\78\\00\\00\\00\\00\\00\\00\\00\\00")
'''
 for name,body in operations.items():
  text+=f'''(func(export "canister_query {name}")
   (local $a v128)(local $b v128)(local $r v128)(local $i i32)(local $ptr i32)(local $begin i64)
   (local.set $ptr(i32.const 64))
   (local.set $a(v128.const i32x4 65537 65537 65537 65537))
   (local.set $b(v128.const i32x4 65537 65537 65537 65537))
   (local.set $i(i32.const 10000))
   (local.set $begin(call $counter(i32.const 0)))
   (loop $again
    {body}
    (br_if $again(local.tee $i(i32.sub(local.get $i)(i32.const 1)))))
   (i64.store(i32.const 7)(i64.sub(call $counter(i32.const 0))(local.get $begin)))
   (call $append(i32.const 0)(i32.const 15))(call $reply))
'''
 text+=')\n';p=d/'probe.wat';p.write_text(text)
 subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(p),str(d/'probe.wasm')],check=True)
 (d/'probe.did').write_text('service:{'+''.join(f'{name}:()->(nat64)query;'for name in operations)+'}')
 files=[Path(__file__),p,d/'probe.wasm',d/'probe.did']
 r=dict(module=hashlib.sha256((d/'probe.wasm').read_bytes()).hexdigest(),iterations=10000,operations=operations,
  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},
  measured=False,scope='Instruction metering calibration only. No inference data, no runtime fee modification or goal completion claim.')
 (d/'build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(module=r['module'],methods=len(operations))))
if __name__=='__main__':main()
