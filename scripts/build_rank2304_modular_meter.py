#!/usr/bin/env python3
"""Local IC core-only benchmark; operands are preprepared, not full inference."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def escaped(b): return ''.join(f'\\{x:02x}' for x in b)
def main():
    source=ROOT/'artifacts/rank2304-modular-simd-v1'
    report=json.loads((source/'execution-report.json').read_text())
    assert report['all24_complete_memory_comparisons_exact']
    for p,h in report['source_hashes'].items(): assert sha(ROOT/p)==h,p
    config=json.loads((source/'execution/config.json').read_text())
    layout=config['layout'];image_bytes=config['cases'][0]['pages']*65536
    # Reserve an extra transposed raw weight buffer for direct SIMD control.
    wt=image_bytes;reply=wt+8192+128;fixture_start=1048576
    wat=(source/'kernel.wat').read_text()
    wat=wat.replace('(module(import "env" "memory" (memory 1))','''(module
 (import "ic0" "performance_counter"(func $pc(param i32)(result i64)))
 (import "ic0" "msg_reply_data_append"(func $append(param i32 i32)))
 (import "ic0" "msg_reply"(func $reply))
 (memory(export "memory") 128)''')
    for mode in ['universal32','checked16']:
        wat=wat.replace(f'(func(export "{mode}")',f'(func ${mode}(export "{mode}")')
    direct='''(func $direct(param $q i32)(param $w i32)(param $out i32)
 (local $t i32)(local $n i32)(local $k i32)(local $v v128)
 (loop $T(local.set $n(i32.const 0))(loop $N
 (local.set $v(v128.const i32x4 0 0 0 0))(local.set $k(i32.const 0))
 (loop $K(local.set $v(i32x4.add(local.get $v)
  (i32x4.dot_i16x8_s
   (v128.load(i32.add(local.get $q)(i32.add(i32.mul(local.get $t)(i32.const 512))(local.get $k))))
   (v128.load(i32.add(local.get $w)(i32.add(i32.mul(local.get $n)(i32.const 512))(local.get $k)))))))
 (local.set $k(i32.add(local.get $k)(i32.const 16)))(br_if $K(i32.lt_u(local.get $k)(i32.const 512))))
 (i32.store(i32.add(local.get $out)(i32.shl(i32.add(i32.mul(local.get $t)(i32.const 16))(local.get $n))(i32.const 2)))
 (i32.add(i32.add(i32x4.extract_lane 0(local.get $v))(i32x4.extract_lane 1(local.get $v)))(i32.add(i32x4.extract_lane 2(local.get $v))(i32x4.extract_lane 3(local.get $v)))))
 (local.set $n(i32.add(local.get $n)(i32.const 1)))(br_if $N(i32.lt_u(local.get $n)(i32.const 16))))
 (local.set $t(i32.add(local.get $t)(i32.const 1)))(br_if $T(i32.lt_u(local.get $t)(i32.const 16)))))
'''
    # Candid (nat64, blob): type vec nat8, two arguments, fixed blob length 1028.
    header=bytes.fromhex('4449444c016d7b027800')
    header+=b'\0'*8+bytes([0x84,0x08])
    body=[direct,f'(data(i32.const {reply})"{escaped(header)}")']
    methods=[]
    for i in range(12):
        image=(source/f'execution/{i}-universal32.input.bin').read_bytes()
        w=np.frombuffer(image,dtype='<i2',count=4096,offset=layout['raww'][0]).reshape(256,16).T.copy()
        fixture=image+w.tobytes();pos=fixture_start+i*len(fixture)
        assert pos+len(fixture)<128*65536
        body.append(f'(data(i32.const {pos})"{escaped(fixture)}")')
        for mode in ['direct','universal32','checked16']:
            name=f'{mode}_{i}'
            if mode=='direct':
                call=f'(call $direct(i32.const {layout["rawq"][0]})(i32.const {wt})(i32.const {layout["out"][0]}))'
            else:
                call=f'(call ${mode}'+''.join(f'(i32.const {layout[n][0]})' for n in ['a','b','p','inner','outer','out','rawq','raww','flag'])+')'
            body.append(f'''(func(export "canister_update {name}")(local $start i64)(local $used i64)
 (memory.copy(i32.const 0)(i32.const {pos})(i32.const {len(fixture)}))
 (local.set $start(call $pc(i32.const 0)))
 {call}
 (local.set $used(i64.sub(call $pc(i32.const 0))(local.get $start)))
 (i64.store(i32.const {reply+10})(local.get $used))
 (call $append(i32.const {reply})(i32.const {len(header)}))
 (call $append(i32.const {layout['out'][0]})(i32.const 1024))
 (call $append(i32.const {layout['flag'][0]})(i32.const 4))
 (call $reply))''')
            methods.append(dict(index=i,mode=mode,method=name))
    assert wat.endswith(')\n')
    wat=wat[:-2]+'\n'+'\n'.join(body)+'\n)\n'
    d=ROOT/'artifacts/rank2304-modular-meter-v1';d.mkdir(exist_ok=False)
    (d/'probe.wat').write_text(wat)
    subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(d/'probe.wat'),str(d/'probe.wasm')],check=True)
    (d/'probe.did').write_text('service:{\n'+''.join(f"{m['method']}:()->(nat64,blob);\n" for m in methods)+'}\n')
    files=[Path(__file__),source/'execution-report.json',source/'kernel.wat',d/'probe.wat',d/'probe.wasm',d/'probe.did']
    (d/'build.json').write_text(json.dumps(dict(complete=True,methods=methods,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},scope='Core only, all input preparation excluded from measurement. Direct raw SIMD is a screening control, not the current best rank7 full projection. No full paid inference claim.'),indent=2)+'\n')
    print('built core-only local meter')
if __name__=='__main__': main()
