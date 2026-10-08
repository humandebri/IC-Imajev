#!/usr/bin/env python3
"""Build an after-reply diagnostic; retain the public paid Candid schema."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from build_paid_message_checkpoint import projection_bodies

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parent = ROOT / 'artifacts/paid-message-checkpoint-v1/build'
    base = json.loads((parent / 'report.json').read_text())
    for key in ['source_hashes', 'dependency_hashes']:
        for p, h in base[key].items():
            assert sha(ROOT / p) == h, p
    assert sha(parent / 'full.wasm') == base['module']
    d = ROOT / 'artifacts/paid-wrapper-checkpoint-v1/build'
    d.mkdir(parents=True, exist_ok=False)
    for p in parent.glob('*.rs'):
        shutil.copyfile(p, d / p.name)
    p = d / 'paid_inference.rs'
    old = p.read_text()
    anchor = '   Ok(v) if v.job_id==id && v.stage>stage && v.stage<=64 => {\n'
    assert old.count(anchor) == 1
    callback = '''    let measured=unsafe{__imajev_worker_checkpoint()};
    PAID.with(|s|{let mut s=s.borrow_mut();
     if let Some(j)=s.active.as_mut(){if j.id==id && j.stage==v.stage{if let Some(m)=j.workers.last_mut(){m.instructions=measured;}}}
     if let Some(r)=s.receipts.iter_mut().find(|r|r.job_id==id){if let ReceiptState::Completed(result)=&mut r.state{if let Some(m)=result.workers.last_mut(){m.instructions=measured;}}}
    });
'''
    new = old.replace(anchor, anchor + callback)
    anchor2 = ' let measured=ic_cdk::api::performance_counter(0);'
    assert new.count(anchor2) == 1
    new = new.replace(anchor2, ' unsafe{__imajev_arm_worker_checkpoint();}\n' + anchor2)
    stubs = '''
// Patched, strictly typed Wasm diagnostic stubs. The getter reads the checkpoint
// saved after the synchronous CDK handler returns, including its Candid reply.
static mut WORKER_CHECKPOINT_STUB:u64=0;
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn __imajev_worker_checkpoint()->u64{std::ptr::read_volatile(std::ptr::addr_of!(WORKER_CHECKPOINT_STUB))}
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn __imajev_arm_worker_checkpoint(){std::ptr::write_volatile(std::ptr::addr_of_mut!(WORKER_CHECKPOINT_STUB),1);}
'''
    new += stubs
    assert new.removesuffix(stubs).replace(callback, '').replace(' unsafe{__imajev_arm_worker_checkpoint();}\n', '') == old
    p.write_text(new)
    command = base['command'][:]
    command[command.index('--edition=2021')+1] = str(d / 'lib.rs')
    command[command.index('-o')+1] = str(d / 'raw.wasm')
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(d), CARGO_PKG_NAME='imajev-inference', CARGO_PKG_VERSION='0.1.0', CARGO_PKG_VERSION_MAJOR='0', CARGO_PKG_VERSION_MINOR='1', CARGO_PKG_VERSION_PATCH='0', CARGO_PKG_VERSION_PRE='', CARGO_CRATE_NAME='imajev_inference')
    with (d / 'compiler.log').open('w') as log:
        subprocess.run(command, env=env, check=True, stdout=log, stderr=log)
    previous = d / 'raw.wasm'
    patches = []
    for i, (entry, donor) in enumerate(zip(base['patches'], projection_bodies((parent / 'full.wasm').read_bytes(), base['patches']), strict=True)):
        source = d / f'donor-{i}.wasm'; source.write_bytes(donor)
        target = d / f'patched-{i}.wasm'
        row = json.loads(subprocess.check_output([str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'), str(previous), str(source), str(target), entry['export']], text=True))
        assert row['wasmparser_validation'] and row['replacement_body_sha256'] == entry['replacement_body_sha256']
        patches.append(row); previous = target
    tool = ROOT / 'artifacts/worker-wrapper-checkpoint-tool-v1/tool'
    wrapper = json.loads(subprocess.check_output([str(tool), str(previous), str(d / 'full.wasm')], text=True))
    files = [Path(__file__), ROOT / 'scripts/build_paid_message_checkpoint.py', ROOT / 'scripts/wasm_worker_checkpoint.rs', tool, parent / 'report.json', parent / 'full.wasm'] + list(d.glob('*.rs'))
    report = dict(complete=True, module=sha(d / 'full.wasm'), parent=base['module'], command=command, patches=patches, wrapper=wrapper, all34_projection_bodies_equal_to_validated_parent=True, scope='Diagnostic after-CDK worker checkpoint with guarded arm and callback metric transfer. Input-dependent infer/transport/callback costs and recording tail require further accounting. No installation/full paid proof/adoption/goal achievement.', full_paid_goal_achieved=False, source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files}, dependency_hashes=base['dependency_hashes'])
    (d / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(module=report['module'], wrapper=wrapper)), flush=True)


if __name__ == '__main__':
    main()
