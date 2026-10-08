#!/usr/bin/env python3
"""Build 512-token paid inference retaining validated optimized kernels."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from build_paid_message_checkpoint import projection_bodies, sha
ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--adaptive-token-tiles', action='store_true', default=True)
    ap.add_argument('--fixed-token-tiles', dest='adaptive_token_tiles', action='store_false',
                    help='Keep the former 57-token Delta/MLP schedule for comparison')
    ap.add_argument('--diagnostics', action='store_true', help='Include owner fault/probe APIs for local verification')
    a = ap.parse_args()
    source = a.source_root.resolve()
    parent = source / 'artifacts/common-prefix27/latest-public-query-build-v2'
    runtime_parent = source / 'artifacts/update-rank7-prepare8-unrolled-v1/build'
    base = json.loads((parent / 'report.json').read_text())
    rt = json.loads((runtime_parent / 'report.json').read_text())
    assert sha(parent / 'full.wasm') == base['wasm_sha256']
    for group in ['source_hashes', 'dependency_hashes']:
        for name, digest in rt[group].items():
            assert sha(source / name) == digest, name
    d = a.directory.resolve()
    d.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), d / 'build-script.py')
    shutil.copytree(runtime_parent / 'runtime', d / 'runtime')
    canonical = ROOT / 'crates/imajev-runtime/src'
    shutil.copyfile(canonical / 'attention_token_chunks.rs', d / 'runtime/attention_token_chunks.rs')
    # Keep frozen projection reuse/SIMD code; insert only the typed state carry.
    path = d / 'runtime/delta_full_log.rs'
    text = path.read_text()
    live = (canonical / 'delta_full_log.rs').read_text()
    begin = live.index(' evaluate_retaining_state(r,x,m,read,initial,None)')
    end = live.index(' if r.op!=', begin)
    old = 'where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {\n if r.op!="delta_full_log_integer"'
    text = replace_once(text, old, 'where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {\n' + live[begin:end] + ' if r.op!="delta_full_log_integer"')
    old = 'let y=if key_major {crate::delta_from_key_major(&qh,&kh,&vh,&gh,&bh,state,128,128)?}else{crate::delta_without_final_state(&qh,&kh,&vh,&gh,&bh,state,128,128)?};'
    new = next(line.strip() for line in live.splitlines() if 'let y=if key_major' in line)
    text = replace_once(text, old, new)
    text = replace_once(text, 'let y=crate::delta_without_final_state(&qh,&kh,&vh,&gh,&bh,state,128,128)?;',
                        'let y=if retained.is_some(){crate::delta(&qh,&kh,&vh,&gh,&bh,state,128,128)?}else{crate::delta_without_final_state(&qh,&kh,&vh,&gh,&bh,state,128,128)?};')
    begin = live.index(' if let Some(slot)=retained')
    end = live.index('\n}', begin)
    text = replace_once(text, '}Ok((out,bytes))', '}\n' + live[begin:end])
    path.write_text(text)
    live = (canonical / 'delta_hybrid.rs').read_text()
    block = live[live.index('/// A job owns'):live.index('impl PreparedDeltaHybrid')]
    path = d / 'runtime/delta_hybrid.rs'
    path.write_text(path.read_text() + '\n' + block + '''
#[cfg(feature="experimental-update-token-chunks")]
impl ServerDeltaPrefix {
 pub fn stream(&self)->ServerDeltaStream {
  ServerDeltaStream{bound:self.bound.clone(),state:Some(crate::delta_full_log::InitialState::KeyMajor(self.state.as_ref().clone())),next:0}
 }
}
''')
    path = d / 'runtime/lib.rs'
    live = (canonical / 'lib.rs').read_text()
    begin = live.index('#[cfg(feature="experimental-update-token-chunks")]\npub use delta_hybrid::ServerDeltaStream;')
    path.write_text(path.read_text() + '\n' + live[begin:])
    for p in parent.glob('*.rs'):
        shutil.copyfile(p, d / p.name)
    for name in ['paid_inference.rs', 'paid_types.rs', 'chunked_update.rs', 'token_plan.rs']:
        shutil.copyfile(ROOT / 'canisters/inference/src' / name, d / name)
    path = d / 'lib.rs'
    path.write_text(replace_once(path.read_text(), 'mod update_inference;', 'mod update_inference;\n#[cfg(feature="experimental-update-token-chunks")]\nmod chunked_update;\n#[cfg(feature="experimental-update-token-chunks")]\nmod token_plan;'))
    path = d / 'update_inference.rs'
    text = replace_once(path.read_text(), 'const STOP: u64 = 34_000_000_000;', 'pub(super) const WORKER_BUDGET: u64 = 30_000_000_000;').replace('<STOP', '<WORKER_BUDGET')
    text = replace_once(text, 'pub(super) fn start(ids: Vec<u32>, options: Vec<String>) -> Result<UpdateProgress,String> {',
                        'pub(super) fn start(ids: Vec<u32>, options: Vec<String>) -> Result<UpdateProgress,String> {\n    if busy(){return Err("inference already active".into());}\n    if ids.len()>89 {return crate::chunked_update::start(ids,options);}')
    text = replace_once(text, 'pub(super) fn continue_graph(id:u64,stage:u64) -> Result<UpdateProgress,String> {',
                        'pub(super) fn continue_graph(id:u64,stage:u64) -> Result<UpdateProgress,String> {\n    if crate::chunked_update::busy(){return crate::chunked_update::continue_graph(id,stage);}')
    text = replace_once(text, 'pub(super) fn busy()->bool {GRAPH.with', 'pub(super) fn busy()->bool {crate::chunked_update::busy() || GRAPH.with')
    text = replace_once(text, 'pub(super) fn release(){GRAPH.with', 'pub(super) fn release(){crate::chunked_update::release();GRAPH.with')
    text += '''
pub(super) fn progress_limit(n:usize)->u64 {if n>89 {crate::token_plan::stages(n)}else{64}}
pub(super) fn stream_prefix(layer:usize)->Result<(Vec<f32>,Option<imajev_runtime::ServerDeltaStream>),String> {
 GRAPH.with(|g|{let g=g.borrow();let b=g.banks.get(g.selected).ok_or("prefix bank missing")?;
  if b.tokens!=27 {return Err("stream prefix identity".into());}
  let p=b.prefix.get(layer).ok_or("prefix layer missing")?;
  let stream=if layer%4==3 {None}else{Some(p.prepared.as_ref().ok_or("prefix state missing")?.stream())};
  Ok((p.values.clone(),stream))
 })
}
'''
    path.write_text(text)
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(d), CARGO_PKG_NAME='imajev-inference', CARGO_PKG_VERSION='0.1.0',
               CARGO_PKG_VERSION_MAJOR='0', CARGO_PKG_VERSION_MINOR='1', CARGO_PKG_VERSION_PATCH='0',
               CARGO_PKG_VERSION_PRE='', CARGO_CRATE_NAME='imajev_inference')
    runtime_command = rt['runtime_command'][:]
    runtime_command[runtime_command.index('--edition=2021') + 1] = str(d / 'runtime/lib.rs')
    runtime_command[runtime_command.index('-o') + 1] = str(d / 'libimajev_runtime.rlib')
    runtime_command += ['--cfg', 'feature="experimental-update-token-chunks"']
    command = base['command'][:]
    command[command.index('--edition=2021') + 1] = str(d / 'lib.rs')
    command[command.index('-o') + 1] = str(d / 'raw.wasm')
    for i, arg in enumerate(command):
        if arg.startswith('imajev_runtime='):
            command[i] = 'imajev_runtime=' + str(d / 'libimajev_runtime.rlib')
    command += ['--cfg', 'feature="experimental-update-token-chunks"']
    if a.adaptive_token_tiles:
        runtime_command += ['--cfg', 'feature="experimental-adaptive-token-tiles"']
        command += ['--cfg', 'feature="experimental-adaptive-token-tiles"']
    if a.diagnostics:
        command += ['--cfg', 'feature="paid-update-diagnostics"']
    files = list(d.glob('*.rs')) + list((d / 'runtime').glob('*.rs')) + [d / 'build-script.py']
    hashes = {str(p): sha(p) for p in files}
    with (d / 'compiler.log').open('w') as log:
        for cmd in [runtime_command, command]:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=log, stderr=log)
    previous = d / 'raw.wasm'
    patches = []
    for i, (entry, donor) in enumerate(zip(base['patches'], projection_bodies((parent / 'full.wasm').read_bytes(), base['patches']), strict=True)):
        donor_path = d / f'donor-{i}.wasm'
        donor_path.write_bytes(donor)
        output = d / ('full.wasm' if i == len(base['patches']) - 1 else f'patched-{i}.wasm')
        row = json.loads(subprocess.check_output([str(source / 'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),
                       str(previous), str(donor_path), str(output), entry['export']], text=True))
        assert row['wasmparser_validation'] and row['replacement_body_sha256'] == entry['replacement_body_sha256']
        patches.append(row)
        previous = output
    assert len(patches) == 34
    assert hashes == {p: sha(Path(p)) for p in hashes}
    report = dict(module=sha(d / 'full.wasm'), runtime_command=runtime_command, command=command, patches=patches,
                  sources=hashes, parent=base['wasm_sha256'], dependencies=rt['dependency_hashes'],
                  local_only=True, input_limit=512, token_chunk=57, attention_group_heads=8,
                  adaptive_token_tiles=a.adaptive_token_tiles, dense_token_chunk=89 if a.adaptive_token_tiles else 57,
                  diagnostics=a.diagnostics,
                  extra_attention_groups_only_above_256=True, all34_projection_bodies_equal=True)
    (d / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(module=report['module'], patches=len(patches))), flush=True)


if __name__ == '__main__':
    main()
