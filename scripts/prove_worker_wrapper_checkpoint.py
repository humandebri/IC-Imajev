#!/usr/bin/env python3
"""Prove after-reply checkpoint persistence on the guarded local small probe."""
import json
import subprocess
from pathlib import Path
from check_sha256_four_lane_probe import ROOT, CID, OWNER, cmd, status, sha


def main():
    d = ROOT / 'artifacts/worker-wrapper-checkpoint-ic-v1'
    d.mkdir(exist_ok=False)
    tools = ROOT / 'artifacts/worker-wrapper-checkpoint-tool-v1'
    wat = r'''(module
 (import "ic0" "performance_counter" (func $pc (param i32)(result i64)))
 (import "ic0" "msg_reply_data_append" (func $append (param i32 i32)))
 (import "ic0" "msg_reply" (func $reply))
 (memory (export "memory") 1)
 (data (i32.const 0) "DIDL\00\00")
 (data (i32.const 64) "DIDL\00\01\78\00\00\00\00\00\00\00\00")
 (global $mode (mut i32)(i32.const 1))
 (func $arm (export "__imajev_arm_worker_checkpoint") nop)
 (func $get (export "__imajev_worker_checkpoint") (result i64) i64.const 999)
 (func $empty i32.const 0 i32.const 6 call $append call $reply)
 (func (export "canister_update inference_step") (local $n i32)
  global.get $mode i32.const 2 i32.eq if
   i32.const 10000 local.set $n
   loop $again local.get $n i32.const 1 i32.sub local.tee $n br_if $again end
  end
  global.get $mode if call $arm end call $empty)
 (func (export "canister_update disable") i32.const 0 global.set $mode call $empty)
 (func (export "canister_update enable") i32.const 2 global.set $mode call $empty)
 (func (export "canister_query checkpoint")
  i32.const 71 call $get i64.store
  i32.const 64 i32.const 15 call $append call $reply))'''
    (d / 'probe.wat').write_text(wat)
    subprocess.run([str(tools / 'wat'), str(d / 'probe.wat'), str(d / 'raw.wasm')], check=True)
    patch = json.loads(subprocess.check_output([str(tools / 'tool'), str(d / 'raw.wasm'), str(d / 'probe.wasm')], text=True))
    (d / 'patch.json').write_text(json.dumps(patch, indent=2) + '\n')
    did = d / 'probe.did'
    did.write_text('service : { inference_step : () -> (); disable : () -> (); enable : () -> (); checkpoint : () -> (nat64) query; }\n')
    before = status()
    (d / 'pre-status.json').write_text(json.dumps(before, indent=2) + '\n')
    assert before['status'] == 'Stopped' and before['settings']['controllers'] == [OWNER]
    assert before['module_hash'].removeprefix('0x') == 'c8096de1f53128612ca38e16465e4216405df2c61bb298bf6caba9d3e0708fc2'
    full_before = status('4caro-hl777-77775-aaaba-cai')
    assert full_before['module_hash'].removeprefix('0x') == '6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
    rows = []

    def call(method, query=False):
        path = d / f'{len(rows)}-{method}.hex'
        raw = cmd(['call', CID, method, '()', '--candid', str(did), '--output', 'hex'] + (['--query'] if query else []))
        path.write_text(raw)
        data = bytes.fromhex(raw.strip().removeprefix('0x'))
        value = None
        if query:
            assert len(data) == 15 and data[:7] == b'DIDL\0\1\x78'
            value = int.from_bytes(data[7:], 'little')
        else:
            assert data == b'DIDL\0\0'
        rows.append(dict(method=method, query=query, value=value, reply=str(path.relative_to(ROOT)), reply_sha256=sha(path)))
        return value

    try:
        (d / 'install.txt').write_text(cmd(['install', CID, '--mode', 'reinstall', '--wasm', str(d / 'probe.wasm'), '--yes']))
        assert status()['module_hash'].removeprefix('0x') == sha(d / 'probe.wasm')
        (d / 'start.txt').write_text(cmd(['start', CID]))
        assert call('checkpoint', True) == 0
        call('inference_step'); first = call('checkpoint', True); assert first > 0
        call('disable'); call('inference_step'); assert call('checkpoint', True) == first
        call('enable'); call('inference_step'); second = call('checkpoint', True); assert second > first
    finally:
        (d / 'stop.txt').write_text(cmd(['stop', CID]))
        after = status(); (d / 'post-status.json').write_text(json.dumps(after, indent=2) + '\n')
    assert after['status'] == 'Stopped' and after['module_hash'].removeprefix('0x') == sha(d / 'probe.wasm')
    full_after = status('4caro-hl777-77775-aaaba-cai')
    assert full_after['module_hash'] == full_before['module_hash'] and full_after['status'] == full_before['status'] == 'Running'
    for name, value in [('full-pre-status.json', full_before), ('full-post-status.json', full_after)]:
        (d / name).write_text(json.dumps(value, indent=2) + '\n')
    files = [Path(__file__), ROOT / 'scripts/wasm_worker_checkpoint.rs', tools / 'tool', tools / 'wat'] + list(d.iterdir())
    result = dict(complete=True, module=sha(d / 'probe.wasm'), first_counter=first, subsequent_counter=second, after_reply_checkpoint_persists_on_ic=True, unarmed_update_preserves_counter=True, subsequent_armed_worker_records_new_counter=True, own_probe_stopped=True, full_baseline_module_and_status_preserved=True, replies=rows, full_paid_goal_achieved=False, scope='IC persistence/arming experiment only. Full paid candidate and helper/callback/tail upper bound remain unverified.', source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    (d / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes', 'replies']}), flush=True)


if __name__ == '__main__':
    main()
