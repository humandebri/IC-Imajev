#!/usr/bin/env python3
"""Meter actual SHA256 components on the guarded, task-owned local probe."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CID = '4zfnl-5t777-77775-aaadq-cai'
OWNER = 'cibxp-okw3l-gfzvi-p6ltu-23ss3-7tlfz-nk65x-tvhvb-mg56y-b5hkf-eqe'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd(args):
    return subprocess.check_output(['icp', 'canister'] + args + ['--network', 'local', '--identity', 'imajev-local'], text=True)


def status(cid=CID):
    return json.loads(cmd(['status', cid, '--json']))


def main():
    d = ROOT / 'artifacts/sha256-four-lane-meter-v1'
    build = json.loads((d / 'build.json').read_text())
    for p, h in build['source_hashes'].items():
        assert sha(ROOT / p) == h, p
    assert sha(d / 'probe.wasm') == build['wasm_sha256']
    check = d / 'check'
    check.mkdir(exist_ok=False)
    source = check / 'decode.rs'
    source.write_text('''fn main(){
 let p=std::env::args().nth(1).unwrap();let text=std::fs::read_to_string(p).unwrap();let s=text.trim().trim_start_matches("0x");
 let bytes:Vec<u8>=(0..s.len()).step_by(2).map(|i|u8::from_str_radix(&s[i..i+2],16).unwrap()).collect();
 let (v,):(Result<(u64,Vec<String>),String>,)=candid::decode_args(&bytes).unwrap();
 println!("{}",serde_json::to_string(&v.unwrap()).unwrap());
}''')
    deps = ROOT / 'artifacts/update-inference/client-target/release/deps'
    native = ['rustc', '--edition=2021', str(source), '-C', 'opt-level=2', '-C', 'panic=abort', '-C', 'lto=thin', '-o', str(check / 'decode'), '-L', 'dependency=' + str(deps)]
    for name in ['candid', 'serde_json']:
        choices = list(deps.glob('lib' + name + '-*.rlib'))
        assert len(choices) == 1
        native += ['--extern', name + '=' + str(choices[0])]
    subprocess.run(native, check=True)
    before = status()
    (check / 'pre-status.json').write_text(json.dumps(before, indent=2) + '\n')
    old = json.loads((ROOT / 'artifacts/rank7-prepare8-unrolled-probe-v1/report.json').read_text())['module']
    assert before['id'] == CID and before['status'] == 'Stopped'
    assert before['module_hash'].removeprefix('0x') == old
    assert before['settings']['controllers'] == [OWNER]
    assert int(before['cycles'].replace('_', '')) > 1_000_000_000_000
    full_before = status('4caro-hl777-77775-aaaba-cai')
    (check / 'full-pre-status.json').write_text(json.dumps(full_before, indent=2) + '\n')
    assert full_before['module_hash'].removeprefix('0x') == '6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
    rows = []
    try:
        (check / 'install.txt').write_text(cmd(['install', CID, '--mode', 'reinstall', '--wasm', str(d / 'probe.wasm'), '--yes']))
        assert status()['module_hash'].removeprefix('0x') == build['wasm_sha256']
        (check / 'start.txt').write_text(cmd(['start', CID]))
        for i, group in enumerate(build['groups']):
            expected = [sha(ROOT / m['path']) for m in group['messages']]
            for simd in [False, True]:
                p = check / f'{i}-{int(simd)}.hex'
                p.write_text(cmd(['call', CID, 'bench', f'({i} : nat32, {str(simd).lower()})', '--candid', str(d / 'probe.did'), '--output', 'hex']))
                instructions, digests = json.loads(subprocess.check_output([str(check / 'decode'), str(p)], text=True))
                assert digests == expected, (i, simd)
                rows.append(dict(group=i, simd=simd, instructions=instructions, digests=digests, reply=str(p.relative_to(ROOT)), reply_sha256=sha(p)))
                print(json.dumps(dict(group=i, simd=simd, instructions=instructions)), flush=True)
    finally:
        (check / 'stop.txt').write_text(cmd(['stop', CID]))
        after = status()
        (check / 'post-status.json').write_text(json.dumps(after, indent=2) + '\n')
    assert after['status'] == 'Stopped' and after['module_hash'].removeprefix('0x') == build['wasm_sha256']
    full_after = status('4caro-hl777-77775-aaaba-cai')
    assert full_after['module_hash'] == full_before['module_hash'] and full_after['status'] == full_before['status']
    (check / 'full-post-status.json').write_text(json.dumps(full_after, indent=2) + '\n')
    comparisons = []
    for i, group in enumerate(build['groups']):
        counts = {r['simd']: r['instructions'] for r in rows if r['group'] == i}
        comparisons.append(dict(group=i, case=group['case'], kind=group['kind'], scalar=counts[False], simd=counts[True], saved=counts[False]-counts[True], percent_saved=(1-counts[True]/counts[False])*100))
    files = [Path(__file__), d / 'build.json'] + sorted(check.iterdir())
    report = dict(complete=True, module=build['wasm_sha256'], replies=rows, comparisons=comparisons, all72_ic_digests_exact=True, own_probe_stopped=True, full_baseline_module_and_status_preserved=True, adopted=False, full_paid_goal_achieved=False, scope=build['scope'], native_command=native, source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    (d / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
