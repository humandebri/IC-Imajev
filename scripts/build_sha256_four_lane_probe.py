#!/usr/bin/env python3
"""Build a local component meter using the paid parent's actual sha2 dependency."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    kernel = ROOT / 'artifacts/sha256-four-lane-kernels-v1'
    execution = json.loads((kernel / 'execution-report.json').read_text())
    assert execution['complete'] and execution['messages'] == 112
    for p, h in execution['source_hashes'].items():
        assert sha(ROOT / p) == h, p
    fixtures = ROOT / 'artifacts/sha256-batch-real-fixtures-v1'
    fr = json.loads((fixtures / 'report.json').read_text())
    for p, h in fr['source_hashes'].items():
        assert sha(ROOT / p) == h, p
    parent = ROOT / 'artifacts/paid-stack-carry-projection-v1/build/report.json'
    base = json.loads(parent.read_text())
    command = base['command']
    flags = []
    dependencies = []
    for i, token in enumerate(command[:-1]):
        if token == '-L':
            flags += ['-L', command[i + 1]]
        if token == '--extern' and command[i + 1].split('=')[0] in {'sha2', 'ic_cdk', 'candid', 'serde'}:
            flags += ['--extern', command[i + 1]]
            dependencies.append(Path(command[i + 1].split('=', 1)[1]))
    assert len(dependencies) == 4
    for p in dependencies:
        assert sha(p) == base['dependency_hashes'][str(p.relative_to(ROOT))]
    d = ROOT / 'artifacts/sha256-four-lane-meter-v1'
    d.mkdir(exist_ok=False)
    literal = '[' + ','.join('[' + ','.join('include_bytes!(' + json.dumps(str(ROOT / m['path'])) + ') as &[u8]' for m in g['messages']) + ']' for g in fr['groups']) + ']'
    rust = '''use sha2::{Digest,Sha256};
static GROUPS:[[&[u8];4];9]=FIXTURES;
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn __imajev_sha256_four(p0:u32,p1:u32,p2:u32,p3:u32,blocks:u32,out:u32,_unused0:u32,_unused1:u32,_unused2:u32){
    for i in 0..32 {std::ptr::write_volatile((out as *mut u32).add(i),p0^p1^p2^p3^blocks);}
}
fn hex(bytes:&[u8])->String{bytes.iter().map(|b|format!("{b:02x}")).collect()}
#[ic_cdk::update]
fn bench(group:u32,simd:bool)->Result<(u64,Vec<String>),String>{
    let raw=GROUPS.get(group as usize).ok_or("group")?;
    let start=ic_cdk::api::performance_counter(0);
    let result=if simd{
        let length=raw[0].len();
        let padded=(length+9+63)/64*64;
        let mut buffers:Vec<Vec<u8>>=Vec::with_capacity(4);
        for message in raw{
            let mut p=vec![0u8;padded];p[..length].copy_from_slice(message);p[length]=128;
            p[padded-8..].copy_from_slice(&((length as u64)*8).to_be_bytes());buffers.push(p);
        }
        let mut words=[0u32;32];
        unsafe{__imajev_sha256_four(buffers[0].as_ptr() as u32,buffers[1].as_ptr() as u32,buffers[2].as_ptr() as u32,buffers[3].as_ptr() as u32,(padded/64) as u32,words.as_mut_ptr() as u32,0,0,0);}
        let mut out=Vec::with_capacity(4);
        for lane in 0..4{let mut bytes=[0u8;32];for word in 0..8{bytes[word*4..word*4+4].copy_from_slice(&words[word*4+lane].to_be_bytes());}out.push(hex(&bytes));}
        drop(buffers);out
    }else{raw.iter().map(|message|hex(&Sha256::digest(message))).collect()};
    let used=ic_cdk::api::performance_counter(0)-start;
    Ok((used,result))
}
'''.replace('FIXTURES', literal)
    source = d / 'probe.rs'
    source.write_text(rust)
    raw = d / 'raw.wasm'
    cmd = ['rustc', '--crate-name', 'sha256_batch_meter', '--edition=2021', str(source), '--crate-type', 'cdylib', '--target', 'wasm32-unknown-unknown', '-C', 'opt-level=3', '-C', 'panic=abort', '-C', 'codegen-units=1', '-C', 'lto=thin', '-C', 'overflow-checks=yes', '-o', str(raw)] + flags
    subprocess.run(cmd, check=True)
    wat = d / 'sha256-four.wat'
    body = (kernel / 'sha256-four.wat').read_text().replace('(export "sha256_four")', '(export "__imajev_sha256_four")')
    # Preserve the existing patcher's strict nine-i32 contract. The extra
    # arguments are unused, and their call overhead stays in the measured window.
    before = '(param $blocks i32)(param $out i32)'
    assert body.count(before) == 1
    body = body.replace(before, before + '(param $unused0 i32)(param $unused1 i32)(param $unused2 i32)')
    wat.write_text(body)
    final = d / 'probe.wasm'
    patch = json.loads(subprocess.check_output([str(ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'), str(raw), str(wat), str(final), '__imajev_sha256_four'], text=True))
    assert patch['wasmparser_validation']
    did = d / 'probe.did'
    did.write_text('service : { bench : (nat32, bool) -> (variant { Ok : record { nat64; vec text }; Err : text }); }\n')
    files = [Path(__file__), parent, kernel / 'execution-report.json', fixtures / 'report.json', source, raw, wat, final, did] + dependencies + [ROOT / m['path'] for g in fr['groups'] for m in g['messages']]
    result = dict(complete=True, wasm_sha256=sha(final), command=cmd, patch=patch, groups=fr['groups'], source_hashes={str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(files)}, scope='Nine real same-length four-message groups. Actual current paid sha2 dependency versus ordinary SIMD. Measured window includes SIMD padding allocation/copy/bit length, compression, lane extraction, hex formatting and buffer freeing; scalar hash and identical hex formatting. Existing raw input buffers and Candid reply/worker wrapper excluded. No installation, IC timing, full inference gain or adoption yet.')
    (d / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Built real-input SHA256 local component meter; not installed')


if __name__ == '__main__':
    main()
